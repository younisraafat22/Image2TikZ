from queue import Queue
from typing import Optional

from transformers import StoppingCriteria
from transformers.generation import streamers

class ExplicitAbort(StoppingCriteria):
    """
    Abort a model generation explicitly (i.e., when using a streamer in a thread).
    """
    def __init__(self):
        super().__init__()
        self.should_stop = False

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self.should_stop

    def reset(self):
        self.should_stop = False
        return self

    def abort(self):
        self.should_stop = True

class TokenStreamer(streamers.BaseStreamer):
    """
    Stream raw token ids (i.e., not decoded strings).
    """
    def __init__(self, skip_prompt: bool = True, timeout: Optional[float] = None):
        self.skip_prompt = skip_prompt
        self.next_tokens_are_prompt = True
        self.token_queue = Queue()
        self.stop_signal = None
        self.timeout = timeout

    def put(self, value):
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TokenStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        for token_id in value.tolist():
            self.token_queue.put(token_id, timeout=self.timeout)

    def end(self):
        self.next_tokens_are_prompt = True
        self.token_queue.put(self.stop_signal, timeout=self.timeout)

    def propagate_error(self, exc):
        self.token_queue.put(exc, timeout=self.timeout)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.token_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration()
        elif isinstance(value, BaseException):
            raise value
        else:
            return value

class TextIteratorStreamer(streamers.TextIteratorStreamer):
    def propagate_error(self, exc):
        self.text_queue.put(exc, timeout=self.timeout)

    def __next__(self):
        value = self.text_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration()
        elif isinstance(value, BaseException):
            raise value
        else:
            return value

class StreamerList(list, streamers.BaseStreamer):
    """
    Similar to StoppingCriteriaList, only for Streamers.
    """
    def put(self, value):
        for streamer in self:
            streamer.put(value)

    def end(self):
        for streamer in self:
            streamer.end()
