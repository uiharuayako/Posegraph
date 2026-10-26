"""Experimental thread concurrency over one explicitly shareable model owner.

The owner is responsible for request-local model state and device scopes. This
pool only coordinates load, bounded admission, delayed unload, and shutdown.
It never makes a legacy mutable estimator thread-safe by wrapping it in threads.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Callable

from .i18n import _lang, set_language, tr
from .inference_pool import InferenceFailure, estimator_info


def create_shared_estimator(config: dict):
    # Keep importing this module possible without Torch, weights, or CUDA.
    from .estimators.shared_sam3d import create_shared_estimator as create
    return create(config)


@contextmanager
def _language(language: str):
    token = _lang.set(_lang.get())
    try:
        set_language(language)
        yield
    finally:
        _lang.reset(token)


class SharedInferencePool:
    """At most N inferences, one owner/model, no child processes.

    release_after=0 unloads once the last active request finishes. Any such
    request marks the current active group for release; concurrent requests
    cannot unload weights underneath one another. Positive idle timeouts remain
    the service's responsibility, as with the process-based InferencePool.
    """

    def __init__(self, parallelism: int, config: dict, factory: Callable = create_shared_estimator):
        if isinstance(parallelism, bool) or not isinstance(parallelism, int) or not 1 <= parallelism <= 8:
            raise ValueError("parallelism must be an integer in 1..8")
        self.parallelism = parallelism
        self.config = dict(config)
        self.factory = factory
        self._condition = threading.Condition(threading.RLock())
        self._owner = None
        self._active = 0
        self._loading = False
        self._changing = False
        self._release_pending = False
        self._closing = False
        self._closed = False
        self._error: str | None = None

    def _state(self) -> dict:
        if self._owner is None:
            info = {"loaded": False, "loading": self._loading, "error": self._error,
                    "device": None, "precision": None, "device_reason": ("",)}
        else:
            info = estimator_info(self._owner, loading=self._loading, error=self._error)
        info.update(strategy="shared", modelCopies=1, loadedModelCopies=int(info["loaded"]),
                    active=self._active, parallelism=self.parallelism, closing=self._closing)
        return info

    @property
    def states(self) -> list[dict]:
        with self._condition:
            return [self._state()]

    @property
    def processes(self) -> list:
        return []

    def diagnostics(self, *, include_pointers=False) -> dict:
        with self._condition:
            info, owner = self._state(), self._owner
        inspect = getattr(owner, "diagnostics", None)
        if callable(inspect):
            info["owner"] = inspect(include_pointers=include_pointers)
        return info

    def _obtain(self, *, reserve: bool):
        while True:
            with self._condition:
                if self._closing:
                    raise RuntimeError("Recognition pool is closed")
                if self._loading or self._changing or (reserve and self._active >= self.parallelism):
                    self._condition.wait()
                    continue
                if self._owner is not None and self._owner.loaded:
                    if reserve:
                        self._active += 1
                    return self._owner
                # Only this thread may create/load. Load/unload never overlaps
                # admitted estimates, and close waits for this transition too.
                if self._active:
                    raise InferenceFailure(503, "Shared recognition model became unavailable during inference")
                self._loading = True
            self._load_owner()

    def _load_owner(self):
        failure = None
        try:
            owner = self._owner
            if owner is None:
                owner = self.factory(dict(self.config))
                with self._condition:
                    self._owner = owner
            owner.load()
            if not owner.loaded:
                raise RuntimeError("Shared model load completed without a loaded model")
        except BaseException as error:
            failure = error
            if self._owner is not None:
                try:
                    self._owner.unload()
                except Exception:
                    pass  # Keep the original loading failure as the diagnostic.
        finally:
            message = None if failure is None else tr("模型加载失败：{error}",
                error=f"{type(failure).__name__}: {failure}")
            with self._condition:
                self._error = message
                self._loading = False
                self._condition.notify_all()
        if failure is not None:
            if isinstance(failure, Exception):
                raise InferenceFailure(503, message) from failure
            raise failure

    def estimate(self, rgb, release_after=-1, language="zh"):
        with _language(language):
            owner = self._obtain(reserve=True)
            try:
                return owner.estimate(rgb)
            except InferenceFailure:
                raise
            except Exception as error:
                # A request failure frees its admission slot, leaving the shared
                # model available. Only lifecycle failures become health errors.
                raise InferenceFailure(500, f"{type(error).__name__}: {error}") from error
            finally:
                self._release(owner, release_after)

    def _release(self, owner, release_after):
        unload = False
        with self._condition:
            self._active -= 1
            self._release_pending |= release_after == 0
            if not self._active and self._release_pending and not self._closing:
                self._changing = True
                self._release_pending = False
                unload = True
            self._condition.notify_all()
        if unload:
            try:
                owner.unload()
            except Exception as error:
                with self._condition:
                    self._error = f"{type(error).__name__}: {error}"
                raise InferenceFailure(500, f"Shared model unload failed: {error}") from error
            finally:
                with self._condition:
                    self._changing = False
                    self._condition.notify_all()

    def preload(self, language="zh"):
        with _language(language):
            self._obtain(reserve=False)

    def close(self):
        with self._condition:
            self._closing = True
            self._condition.notify_all()  # Rejected queued requests need not await running ones.
            self._condition.wait_for(lambda: not self._active and not self._loading and not self._changing)
            if self._closed:
                return
            self._changing = True
            owner = self._owner
        success = False
        try:
            if owner is not None:
                owner.unload()
            success = True
        except Exception as error:
            with self._condition:
                self._error = f"{type(error).__name__}: {error}"
            raise InferenceFailure(500, f"Shared model shutdown failed: {error}") from error
        finally:
            with self._condition:
                if success:
                    self._closed = True
                    self._owner = None
                    self._release_pending = False
                self._changing = False
                self._condition.notify_all()
