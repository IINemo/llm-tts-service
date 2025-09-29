import openai
import os
import sys
import time
import diskcache as dc
import threading

import logging

log = logging.getLogger()


class DeepSeekChat:
    def __init__(
            self,
            cache_path: str,
            api_base: str | None = "https://api.deepseek.com/v1",
            model: str = "deepseek-reasoner",
            api_key: str | None = None,
            wait_times: tuple = (5, 10, 30, 60, 120),
    ):
        if api_key is None:
            api_key = os.environ.get("DEEPSEEK_API_KEY", None)
            
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        if cache_path is None:
            cache_path = '~/.cache'
        self.cache_path = os.path.join(cache_path, "deepseek_chat_cache.diskcache")
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        self.wait_times = wait_times
        
        # Initialize cache with proper settings
        cache_settings = dc.DEFAULT_SETTINGS.copy()
        cache_settings["eviction_policy"] = "none"
        cache_settings["size_limit"] = int(1e12)
        cache_settings["cull_limit"] = 0
        self.cache = dc.Cache(self.cache_path, **cache_settings)
        self._lock = threading.Lock()

    def ask(self, message: str, json_output=False) -> str:
        # First try to get from cache without lock
        reply = self.cache.get((self.model, message), '')
        
        if reply == '':
            if self.api_key is None:
                raise Exception("Cant ask DeepSeek without token.")
            messages = [
                {"role": "user", "content": message},
            ]
            chat = self._send_request(messages, json_output)
            if chat is None:
                reply = ""
            else:
                reply = chat.choices[0].message.content
            # Only lock when writing to cache
            with self._lock:
                self.cache[(self.model, message)] = reply

        if any(x in reply.lower() for x in ["please provide", "to assist you", "as an ai language model"]):
            return ""

        return reply

    def _send_request(self, messages, json_output=False):
        chat_args = {
            'model': self.model,
            'messages': messages,
            'temperature': 0.6,
        }
        if json_output:
            chat_args['response_format'] = {'type': 'json_object'}
        for i in range(len(self.wait_times)):
            try:
                return openai.OpenAI(base_url=self.api_base, api_key=self.api_key).chat.completions.create(**chat_args)
            except Exception as e:
                sleep_time = self.wait_times[i]
                log.warning(
                    f"Request failed with exception: {e}. Retry #{i}/5 after {sleep_time} seconds."
                )
                time.sleep(sleep_time)
        try:
            return openai.OpenAI(base_url=self.api_base, api_key=self.api_key).chat.completions.create(**chat_args)
        except Exception as e:
            sys.stderr.write(f'Error: {e}')
            return None

    def __del__(self):
        """Cleanup method to properly close the cache when the instance is destroyed."""
        if hasattr(self, 'cache'):
            self.cache.close()