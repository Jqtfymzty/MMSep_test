"""Tests for auto_eval.py.

All tests use local data and fake Ark responses; no API key or network is needed.
Run with: pytest -q test_auto_eval.py
"""

import importlib
import json
import shutil
import sys
import types
from pathlib import Path
from uuid import uuid4

import pytest

# pytest 的系统临时目录在部分 Windows 环境中没有写权限，因此测试使用项目内目录。


@pytest.fixture
def local_tmp_path():
    """Use a project-local temporary directory on restricted Windows accounts."""
    root = Path(__file__).resolve().parent / ".pytest_runtime"
    path = root / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


## 导入被测模块：缺少 Ark SDK 时注入替身，测试不依赖网络或 API Key。
def _load_auto_eval():
    """Allow the unit tests to run even when the optional Ark SDK is absent."""
    try:
        return importlib.import_module("auto_eval")
    except ModuleNotFoundError as exc:
        if exc.name != "volcenginesdkarkruntime":
            raise
        fake_sdk = types.ModuleType("volcenginesdkarkruntime")

        class Ark:
            pass

        fake_sdk.Ark = Ark
        sys.modules["volcenginesdkarkruntime"] = fake_sdk
        return importlib.import_module("auto_eval")


auto_eval = _load_auto_eval()


## 数据分块测试：覆盖正常输入和边界参数。
def test_split_list_evenly_enough():
    assert auto_eval.split_list(list(range(7)), 3) == [[0, 1, 2], [3, 4, 5], [6]]


def test_split_list_empty():
    # Current implementation raises ValueError because chunk_size becomes zero.
    # Keep this regression test until the empty-input behavior is fixed.
    with pytest.raises(ValueError):
        auto_eval.split_list([], 3)


def test_split_list_rejects_zero_chunks():
    with pytest.raises(ZeroDivisionError):
        auto_eval.split_list([1], 0)


def test_get_chunk_returns_requested_chunk():
    assert auto_eval.get_chunk(list(range(6)), 2, 1) == [3, 4, 5]


def test_get_chunk_rejects_invalid_index():
    with pytest.raises(IndexError):
        auto_eval.get_chunk([1, 2], 2, 2)


## 文件读取测试：使用临时 JSONL，验证正常读取和文件不存在异常。
def test_load_reference_and_answer_reads_jsonl(local_tmp_path):
    reference = local_tmp_path / "reference.jsonl"
    answer = local_tmp_path / "answer.jsonl"
    reference.write_text(
        '{"question_id":"q1","question":"What?","answer":"A"}\n'
        '{"question_id":"q2","question":"Why?","answer":"B"}\n',
        encoding="utf-8",
    )
    answer.write_text(
        '{"question_id":"q1","text":"Model A"}\n'
        '{"question_id":"q2","text":"Model B"}\n',
        encoding="utf-8",
    )

    refs, answers = auto_eval.load_reference_and_answer(str(reference), str(answer), 2, 1)
    assert refs == [{"question_id": "q2", "question": "Why?", "answer": "B"}]
    assert answers == [{"question_id": "q2", "text": "Model B"}]


def test_load_reference_and_answer_reports_missing_file(local_tmp_path):
    with pytest.raises(FileNotFoundError):
        auto_eval.load_reference_and_answer(
            str(local_tmp_path / "missing.jsonl"), str(local_tmp_path / "answer.jsonl"), 1, 0
        )


def test_doubao_prompt_txt_contains_all_answers():
    prompt = auto_eval.doubao_prompt_txt("Question", "Reference", "Model")
    assert "Question" in prompt
    assert "Reference" in prompt
    assert "Model" in prompt
    assert "Assistant 1" in prompt and "Assistant 2" in prompt


def test_doubao_prompt_txt_abs_mode_is_explicitly_unsupported():
    with pytest.raises(NotImplementedError):
        auto_eval.doubao_prompt_txt("Q", "R", "M", mode="abs")


def test_doubao_prompt_img_contains_image_and_question():
    prompt = auto_eval.doubao_prompt_img("Question", "Reference", "Model", "aGVsbG8=")
    assert prompt[0]["type"] == "image_url"
    assert "aGVsbG8=" in prompt[0]["image_url"]["url"]
    assert "Question" in prompt[1]["text"]


## 以下类模拟 Ark SDK 返回对象，避免测试真实网络接口。
class _Response:
    class _Choice:
        class _Message:
            content = "8 7\nThe second answer is less detailed."

        message = _Message()

    choices = [_Choice()]


class _SuccessfulCompletions:
    @staticmethod
    def create(**kwargs):
        return _Response()


class _SuccessfulChat:
    completions = _SuccessfulCompletions()


class _Client:
    chat = _SuccessfulChat()


def test_doubao_chat_step_txt_returns_client_content():
    result = auto_eval.doubao_chat_step_txt(_Client(), "prompt")
    assert result.startswith("8 7")


class _FailingCompletions:
    @staticmethod
    def create(**kwargs):
        raise RuntimeError("offline")


class _FailingChat:
    completions = _FailingCompletions()


class _FailingClient:
    chat = _FailingChat()


def test_doubao_chat_step_txt_handles_api_error():
    assert auto_eval.doubao_chat_step_txt(_FailingClient(), "prompt") == "-1 -1\nError during evaluation."


## 评分解析测试：覆盖正常分数、浮点分数和非法响应。
def test_parse_score_relative_float_and_explanation():
    scores, explanation = auto_eval.parse_score("8.5 7\nGood comparison.")
    assert scores == [8, 7]
    assert explanation == "Good comparison."


def test_parse_score_invalid_relative_response():
    scores, explanation = auto_eval.parse_score("not scores")
    assert scores == [-1, -1]
    assert "Error parsing" in explanation


def test_parse_score_absolute_mode():
    scores, explanation = auto_eval.parse_score("9\nClear answer.", mode="abs")
    assert scores == [9]
    assert explanation == "Clear answer."


## 端到端测试：验证读取、调用 Mock、解析评分和写出 JSONL。
def test_eval_doubao_review_writes_jsonl_result(local_tmp_path, monkeypatch):
    reference = local_tmp_path / "reference.jsonl"
    answer = local_tmp_path / "answer.jsonl"
    reference.write_text(
        json.dumps({"question_id": "q1", "question": "What?", "answer": "Reference"}) + "\n",
        encoding="utf-8",
    )
    answer.write_text(
        json.dumps({"question_id": "q1", "text": "Model"}) + "\n",
        encoding="utf-8",
    )

    class FakeArk:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        chat = _SuccessfulChat()

    monkeypatch.setattr(auto_eval, "Ark", FakeArk)
    save_file = local_tmp_path / "results" / "scores.jsonl"
    args = types.SimpleNamespace(
        reference=str(reference),
        answer=str(answer),
        save_file=str(save_file),
        image_folder=str(local_tmp_path),
        scorer="doubao_txt",
        mode="rel",
        max_tokens=32,
        temperature=0.0,
        top_p=1.0,
        num_chunks=1,
        chunk_idx=0,
    )

    auto_eval.eval_doubao_review(args)
    result = json.loads(save_file.read_text(encoding="utf-8").strip())
    assert result["question_id"] == "q1"
    assert result["ref_score"] == 8
    assert result["model_score"] == 7
    assert result["model_ans"] == "Model"
