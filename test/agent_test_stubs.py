import importlib
import sys
import types

import numpy as np


class _DummyPrompt:
    def __init__(self, template, values):
        self.template = template
        self.values = values

    def to_string(self):
        try:
            return self.template.format(**self.values)
        except Exception:
            return self.template


class _DummyPromptTemplate:
    def __init__(self, template):
        self.template = template

    @classmethod
    def from_template(cls, template):
        return cls(template)

    def invoke(self, values):
        return _DummyPrompt(self.template, values)


class _DummyChatOpenAI:
    def __init__(self, *args, **kwargs):
        pass

    def invoke(self, *args, **kwargs):
        return types.SimpleNamespace(content="```python\nprint('ok')\n```")


class _DummyFileLock:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyOpenAI:
    def __init__(self, *args, **kwargs):
        self.responses = types.SimpleNamespace(
            create=lambda **kw: types.SimpleNamespace(output_text="stub_doc")
        )
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(
                create=lambda **kw: types.SimpleNamespace(
                    choices=[
                        types.SimpleNamespace(
                            message=types.SimpleNamespace(content='{"choice": ["IForest"]}')
                        )
                    ]
                )
            )
        )


class _DummyDataLoader:
    def __init__(self, filepath, store_script=True, store_path=None):
        self.filepath = filepath

    def load_data(self, split_data=False):
        return np.zeros((10, 3)), np.zeros(10)



def install_common_stubs():
    langchain_core_prompts = types.ModuleType("langchain_core.prompts")
    langchain_core_prompts.PromptTemplate = _DummyPromptTemplate

    langchain_prompts = types.ModuleType("langchain.prompts")
    langchain_prompts.PromptTemplate = _DummyPromptTemplate

    langchain_openai = types.ModuleType("langchain_openai")
    langchain_openai.ChatOpenAI = _DummyChatOpenAI

    langchain_core_messages = types.ModuleType("langchain_core.messages")
    for name in ["AIMessage", "HumanMessage", "SystemMessage", "BaseMessage"]:
        setattr(
            langchain_core_messages,
            name,
            type(name, (), {"__init__": lambda self, content=None: setattr(self, "content", content)}),
        )

    filelock_mod = types.ModuleType("filelock")
    filelock_mod.FileLock = _DummyFileLock

    openai_mod = types.ModuleType("openai")
    openai_mod.OpenAI = _DummyOpenAI
    openai_mod.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **kw: None))

    data_loader_mod = types.ModuleType("data_loader.data_loader")
    data_loader_mod.DataLoader = _DummyDataLoader

    sys.modules["langchain_core.prompts"] = langchain_core_prompts
    sys.modules["langchain.prompts"] = langchain_prompts
    sys.modules["langchain_openai"] = langchain_openai
    sys.modules["langchain_core.messages"] = langchain_core_messages
    sys.modules["filelock"] = filelock_mod
    sys.modules["openai"] = openai_mod
    sys.modules["data_loader.data_loader"] = data_loader_mod


def load_real_agent_module(module_name: str):
    if "agents" in sys.modules and not hasattr(sys.modules["agents"], "__path__"):
        del sys.modules["agents"]
    if module_name in sys.modules:
        del sys.modules[module_name]
    return importlib.import_module(module_name)
