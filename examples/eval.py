#!/usr/bin/env -S torchrun --nproc_per_node gpu
from argparse import ArgumentParser
from collections import defaultdict
from datetime import timedelta
from functools import partial
from itertools import count
from json import dump, load as load_json
from operator import itemgetter
from os import getenv
from os.path import isfile, join
from time import time

from datasets import load_dataset
import os, sys
# Ensure local repo modules (e.g., detikzify/*) are preferred over site-packages
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Custom TikZ compile temp-dir handling (align with eval_model approach) ---
import shutil
import tempfile
from shutil import which
from subprocess import DEVNULL
from detikzify.infer import TikzDocument
from detikzify.evaluate import ImageSim  # keep imports working
from detikzify.infer.tikz import logger as tikz_logger
from detikzify.util import check_output
from pdfCropMargins import crop
import fitz

# Base temp dir (respect existing TMPDIR; otherwise create a repo-local tmp)
BASE_TIKZ_TMPDIR = os.environ.get("TMPDIR") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_tikz_tmp"))
os.makedirs(BASE_TIKZ_TMPDIR, exist_ok=True)

class _CustomTempDir:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self._counter = 0

    def get(self) -> str:
        self._counter += 1
        d = os.path.join(self.base_dir, f"tikz_eval_{os.getpid()}_{self._counter}")
        os.makedirs(d, exist_ok=True)
        return d

    def cleanup(self, path: str):
        try:
            if os.path.exists(path):
                shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass

_tikz_tmp_mgr = _CustomTempDir(BASE_TIKZ_TMPDIR)

_orig_compile = TikzDocument.compile

def _patched_compile(self: TikzDocument):
    out = dict()
    tmpdir = _tikz_tmp_mgr.get()
    tex_path = None
    try:
        # Write .tex
        import uuid
        stem = os.path.join(tmpdir, f"tikz_{uuid.uuid4().hex[:8]}")
        tex_path = stem + ".tex"
        lines = self.code.split("\n")
        lines.insert(1, r"{cmd}\AtBeginDocument{{{cmd}}}".format(cmd=r"\thispagestyle{empty}\pagestyle{empty}"))
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        # Some classes expect .bbl
        open(stem + ".bbl", "a").close()

        tmppdf = stem + ".pdf"
        outpdf = os.path.join(tmpdir, "tikz.pdf")

        def _save_last_page():
            try:
                doc = fitz.open(tmppdf)
                doc.select([len(doc) - 1])
                doc.save(outpdf)
                doc.close()
            except Exception:
                pass

        # Prefer tectonic; fallback to latexmk -pdf
        if which("tectonic") is not None:
            try:
                check_output(
                    cwd=tmpdir,
                    timeout=self.timeout,
                    stderr=DEVNULL,
                    env=os.environ | dict(max_print_line="1000"),
                    args=["tectonic", "-Z", "shell-escape", "-o", tmpdir, tex_path],
                )
            except Exception as proc:
                log = (getattr(proc, "output", b"") or b"").decode(errors="ignore")
                out.update(status=getattr(proc, "returncode", -1), log=log)
                _save_last_page()
            else:
                out.update(status=0, log="")
                _save_last_page()
        else:
            # latexmk path
            try:
                check_output(
                    cwd=tmpdir,
                    timeout=self.timeout,
                    stderr=DEVNULL,
                    env=os.environ | dict(max_print_line="1000"),
                    args=["latexmk", "-f", "-nobibtex", "-norc", "-file-line-error", "-interaction=nonstopmode", "-pdf", tex_path],
                )
            except Exception as proc:
                log = (getattr(proc, "output", b"") or b"").decode(errors="ignore")
                out.update(status=getattr(proc, "returncode", -1), log=log)
                _save_last_page()
            else:
                out.update(status=0, log="")
                _save_last_page()

        # Crop and return a fitz doc from bytes to detach from files
        croppdf = stem + ".crop"
        try:
            crop(["-gsf", "-c", "gb", "-p", "0", "-a", "-1", "-o", croppdf, outpdf], quiet=True)
        except Exception:
            pass
        target_pdf = croppdf if os.path.isfile(croppdf) else (outpdf if os.path.isfile(outpdf) else None)
        if target_pdf:
            with open(target_pdf, "rb") as pf:
                pdf_bytes = pf.read()
            out["pdf"] = fitz.open(stream=pdf_bytes, filetype="pdf")

    except FileNotFoundError:
        tikz_logger.error("Missing dependencies: install Tectonic or TeX Live tools (ghostscript/poppler).")
    except Exception as e:
        tikz_logger.error(f"TikZ compile error: {e}")
    finally:
        # best-effort cleanup of the temp dir
        _tikz_tmp_mgr.cleanup(tmpdir)

    if out.get("status") == 0 and not out.get("pdf"):
        tikz_logger.warning("Compiled but no PDF produced after cropping")
    return TikzDocument.Output(**out)  # type: ignore

# Apply monkey-patch so this eval uses robust temp-dir compilation
TikzDocument.compile = _patched_compile
from numpy import array
from scipy.stats.mstats import winsorize
from torch import bfloat16, distributed as dist, float16
from torch.cuda import is_available as is_cuda_available, is_bf16_supported
from tqdm import tqdm
from transformers import set_seed
from transformers.utils import is_flash_attn_2_available

from detikzify.evaluate import (
    CrystalBLEU,
    KernelInceptionDistance,
    ImageSim,
    TexEditDistance,
)
from detikzify.infer import DetikzifyPipeline, TikzDocument
from detikzify.model import load as load_model

WORLD_SIZE = int(getenv("WORLD_SIZE", 1))
RANK = int(getenv("RANK", 0))

def parse_args():
    argument_parser = ArgumentParser(
        description="Evaluate fine-tuned models."
    )
    argument_parser.add_argument(
        "--cache_dir",
        help="directory where model outputs should be saved to",
    )
    argument_parser.add_argument(
        "--trainset",
        required=True,
        help="path to the datikz train set (in parquet format)",
    )
    argument_parser.add_argument(
        "--testset",
        required=True,
        help="path to the datikz test set (in parquet format)",
    )
    argument_parser.add_argument(
        "--output",
        required=True,
        help="where to save the computed scores (as json)",
    )
    argument_parser.add_argument(
        "--max_eval_samples",
        type=int,
        help="evaluate on at most this many test samples (for quick runs)",
    )
    argument_parser.add_argument(
        "--timeout",
        type=int,
        help="minimum time to run MCTS in seconds",
    )
    argument_parser.add_argument(
        "--max_no_compile_iters",
        type=int,
        default=10,
        help="stop after this many consecutive iterations without a compilable output (in addition to any timeout)",
    )
    argument_parser.add_argument(
        "--use_sketches",
        action="store_true",
        help="condition model on sketches instead of images",
    )
    argument_parser.add_argument(
        "--path",
        nargs='+',
        metavar="MODEL=PATH",
        required=True,
        help="(multiple) key-value pairs of model names and paths/urls to models/adapters (local or hub) or json files",
    )
    return argument_parser.parse_args()

# https://stackoverflow.com/a/54802737
def chunk(l, n):
    """Yield n number of striped chunks from l."""
    for i in range(0, n):
        yield l[i::n]

def interleave(chunks):
    """Interleave chunks until one is exhausted."""
    interleaved = list()
    for idx in count():
        try:
            interleaved.extend(chunk[idx] for chunk in chunks)
        except IndexError:
            break
    return interleaved

def generate(pipe, image, strict=False, timeout=None, max_no_compile_iters=10, **tqdm_kwargs):
    """Run MCTS until the generated tikz code compiles."""
    start, success, tikzpics = time(), False, set()
    no_compile_streak = 0
    for score, tikzpic in tqdm(pipe.simulate(image=image), desc="Try", **tqdm_kwargs):
        tikzpics.add((score, tikzpic))
        is_compilable = (not tikzpic.compiled_with_errors) if strict else tikzpic.is_rasterizable
        if is_compilable:
            success = True
            no_compile_streak = 0
        else:
            no_compile_streak += 1
        # stop if we reached the no-compile cap
        if max_no_compile_iters is not None and no_compile_streak >= max_no_compile_iters:
            break
        # stop if we reached the time budget after a success
        if success and (not timeout or time() - start >= timeout):
            break
    return [tikzpic for _, tikzpic in sorted(tikzpics, key=itemgetter(0))]

def predict(model_name, base_model, testset, cache_file=None, timeout=None, key="image", max_no_compile_iters=10):
    predictions, worker_preds = list(), list()
    model, tokenizer = load_model(
        base_model=base_model,
        device_map=RANK,
        torch_dtype=bfloat16 if is_cuda_available() and is_bf16_supported() else float16,
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
    )
    # if we don't have a timeout (i.e., only run mcts until we obtain smth compileable), we can use fast metrics
    metric_type = "model" if timeout else "fast"
    pipe = DetikzifyPipeline(model=model, tokenizer=tokenizer, metric=metric_type)
    if cache_file and isfile(cache_file):
        with open(cache_file) as f:
            # disable timeout as we know that the (last) images compile
            predictions = [[TikzDocument(code, timeout=None) for code in sample] for sample in load_json(f)]
    try:
        worker_chunk = list(chunk(list(testset)[len(predictions):], WORLD_SIZE))[RANK]
        # FIXME: right now there only is a progress bar for Rank 0
        for item in tqdm(worker_chunk, desc=f"{model_name.title()} ({RANK})", disable=RANK!=0):
            tikz = generate(
                pipe,
                image=item[key],
                timeout=timeout,
                max_no_compile_iters=max_no_compile_iters,
                position=1,
                leave=False,
                disable=RANK!=0,
            )
            worker_preds.append(tikz)
        del model, tokenizer, pipe
    finally:
        dist.all_gather_object(gathered:=WORLD_SIZE * [None], worker_preds)
        predictions.extend(interleave(gathered))
        if cache_file and RANK == 0:
            with open(cache_file, 'w') as f:
                dump([[p.code for p in ps] for ps in predictions], f)
    return predictions

def load_metrics(trainset, measure_throughput=False, **kwargs):
    bleu = CrystalBLEU(corpus=trainset, **kwargs)
    eed = TexEditDistance(**kwargs)
    emdsim = ImageSim(mode="emd", **kwargs)
    cossim = ImageSim(**kwargs)
    kid = KernelInceptionDistance(**kwargs)
    
    # Import DreamSim
    try:
        from detikzify.evaluate.dreamsim import DreamSim
        dreamsim = DreamSim(**kwargs)
    except Exception as e:
        print(f"[WARN] DreamSim disabled due to import error: {e}")
        dreamsim = None

    def mean_token_efficiency(predictions, limit=0.05):
        samples = list()
        for preds in predictions:
            samples.append(len(preds[-1].code)/sum(len(pred.code) for pred in preds))
        return winsorize(array(samples), limits=limit).mean().item()

    def mean_sampling_throughput(predictions, limit=0.05):
        return winsorize(array(list(map(len, predictions))), limits=limit).mean().item()

    def compute(references, predictions):
        ref_code, pred_code = [[ref['code']] for ref in references], [pred[-1].code for pred in predictions]
        ref_image, pred_image = [ref['image'] for ref in references], [pred[-1].rasterize() for pred in predictions]
        
        # Filter out failed compilations (None images) for visual metrics
        total_samples = len(pred_image)
        valid_pairs = []
        failed_compilations = 0
        
        for ref_img, pred_img in zip(ref_image, pred_image):
            if pred_img is not None:
                valid_pairs.append((ref_img, pred_img))
            else:
                failed_compilations += 1
        
        successful_compilations = len(valid_pairs)
        compilation_success_rate = successful_compilations / total_samples if total_samples > 0 else 0.0
        
        print(f"[INFO] Compilation Stats: {successful_compilations}/{total_samples} successful ({compilation_success_rate:.2%})")
        
        if measure_throughput:
            scores = {"MeanSamplingThroughput": mean_sampling_throughput(predictions=predictions)}
        else:
            scores = {"MeanTokenEfficiency": mean_token_efficiency(predictions=predictions)}
        
        # Add compilation statistics
        scores.update({
            "TotalSamples": total_samples,
            "SuccessfulCompilations": successful_compilations,
            "FailedCompilations": failed_compilations,
            "CompilationSuccessRate": compilation_success_rate
        })

        # Only compute visual metrics if we have valid compilations
        if successful_compilations > 0:
            valid_ref_images, valid_pred_images = zip(*valid_pairs) if valid_pairs else ([], [])
               
            metrics = {
                bleu: partial(bleu.update, list_of_references=ref_code, hypotheses=pred_code),
                eed: partial(eed.update, target=ref_code, preds=pred_code),
                emdsim: lambda: [emdsim.update(img1=img1, img2=img2) for img1, img2 in zip(valid_ref_images, valid_pred_images)],
                cossim: lambda: [cossim.update(img1=img1, img2=img2) for img1, img2 in zip(valid_ref_images, valid_pred_images)],
                kid: lambda: [(kid.update(img1, True), kid.update(img2, False)) for img1, img2 in zip(valid_ref_images, valid_pred_images)],
            }

            # Include DreamSim if available
            if dreamsim is not None:
                metrics[dreamsim] = lambda: [dreamsim.update(img1=img1, img2=img2) for img1, img2 in zip(valid_ref_images, valid_pred_images)]

            for metric, update in metrics.items():
                try:
                    update()
                    scores[str(metric)] = metric.compute() # type: ignore
                    metric.reset()
                except Exception as e:
                    print(f"[ERROR] Failed to compute {metric}: {e}")
                    scores[str(metric)] = None
        else:
            print("[WARN] No successful compilations found - visual metrics will be None")
            scores.update({
                "CrystalBLEU": None,
                "TexEditDistance": None,
                "ImageSim(cosine)": None,
                "ImageSim(emd)": None,
                "KernelInceptionDistance": None,
                "DreamSim": None
            })
            
        return scores

    return compute

if __name__ == "__main__":
    set_seed(0)
    dist.init_process_group(timeout=timedelta(days=3))
    args = parse_args()

    trainset = load_dataset("parquet", data_files=args.trainset, split="train")
    testset = load_dataset("parquet", data_files={"test": args.testset}, split="test").sort("caption") # type: ignore
    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        limit = min(args.max_eval_samples, len(testset))
        testset = testset.select(range(limit))

    predictions = defaultdict(list)
    for model_name, path in map(lambda s: s.split('=', 1), tqdm(args.path, desc="Predicting")):
        if path.endswith("json"):
            with open(path) as f:
                predictions[model_name] = [[TikzDocument(code, None) for code in sample] for sample in load_json(f)]
        else:
            cache_file = join(args.cache_dir, f'{model_name}.json') if args.cache_dir else None
            predictions[model_name] = predict(
                model_name=model_name,
                base_model=path,  # supports local directories or HF hub IDs
                testset=testset,
                cache_file=cache_file,
                timeout=args.timeout,
                key="sketch" if args.use_sketches else "image",
                max_no_compile_iters=args.max_no_compile_iters,
            )

    if RANK == 0: # Scoring only on main process
        scores = dict()
        metrics = load_metrics(trainset['code'], measure_throughput=args.timeout is not None, sync_on_compute=False) # type: ignore
        for model_name, prediction in tqdm(predictions.items(), desc="Computing metrics", total=len(predictions)):
            scores[model_name] = metrics(references=testset, predictions=prediction)
        with open(args.output, "w") as file:
            dump(scores, file)