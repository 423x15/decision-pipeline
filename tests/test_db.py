import pytest
from pathlib import Path
from agent.db import Database


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "test.db")


def test_create_conversation_returns_uuid(db):
    conv_id = db.create_conversation("Test")
    assert isinstance(conv_id, str) and len(conv_id) == 36


def test_list_conversations_empty(db):
    assert db.list_conversations() == []


def test_list_conversations_after_create(db):
    db.create_conversation("Conv 1")
    db.create_conversation("Conv 2")
    convs = db.list_conversations()
    assert len(convs) == 2
    assert all("id" in c and "title" in c and "created_at" in c for c in convs)


def test_save_and_load_messages(db):
    conv_id = db.create_conversation("Test")
    db.save_message(conv_id, "user", "Bonjour", None)
    db.save_message(conv_id, "assistant", "Réponse", [{"type": "tool_called"}])
    msgs = db.load_messages(conv_id)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["tool_trace"] == [{"type": "tool_called"}]


def test_delete_conversation(db):
    conv_id = db.create_conversation("To delete")
    db.save_message(conv_id, "user", "msg", None)
    db.delete_conversation(conv_id)
    assert db.list_conversations() == []
    assert db.load_messages(conv_id) == []
