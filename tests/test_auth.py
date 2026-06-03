import pytest
import pytest_asyncio
import aiosqlite
from vllm_metrics_proxy.db import init_db
from vllm_metrics_proxy.auth import (
    create_api_key,
    get_api_key,
    list_api_keys,
    delete_api_key,
    update_api_key,
    mask_key,
)


@pytest_asyncio.fixture
async def db_path(tmp_path):
    path = str(tmp_path / "test.db")
    await init_db(path)
    return path


@pytest.mark.asyncio
async def test_api_keys_table_created(db_path):
    async with aiosqlite.connect(db_path) as conn:
        async with conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='api_keys'"
        ) as cur:
            row = await cur.fetchone()
            assert row is not None


@pytest.mark.asyncio
async def test_create_api_key_no_expiry(db_path):
    key = await create_api_key(db_path, name="test-key")
    assert key["name"] == "test-key"
    assert key["expired_at"] is None
    assert key["enabled"] == 1
    assert "id" in key
    import uuid
    uuid.UUID(key["id"])


@pytest.mark.asyncio
async def test_create_api_key_with_expiry(db_path):
    key = await create_api_key(db_path, name="temp-key", expires_in="30d")
    assert key["expired_at"] is not None
    assert key["name"] == "temp-key"


@pytest.mark.asyncio
async def test_get_api_key(db_path):
    created = await create_api_key(db_path, name="my-key")
    fetched = await get_api_key(db_path, created["id"])
    assert fetched is not None
    assert fetched["id"] == created["id"]
    assert fetched["name"] == "my-key"


@pytest.mark.asyncio
async def test_get_api_key_not_found(db_path):
    fetched = await get_api_key(db_path, "nonexistent-id")
    assert fetched is None


@pytest.mark.asyncio
async def test_list_api_keys_masked(db_path):
    await create_api_key(db_path, name="key-a")
    await create_api_key(db_path, name="key-b")
    keys = await list_api_keys(db_path)
    assert len(keys) == 2
    for k in keys:
        assert k["masked_id"] == mask_key(k["id"])
        assert k["masked_id"] != k["id"]


@pytest.mark.asyncio
async def test_delete_api_key(db_path):
    created = await create_api_key(db_path, name="to-delete")
    await delete_api_key(db_path, created["id"])
    fetched = await get_api_key(db_path, created["id"])
    assert fetched is None


@pytest.mark.asyncio
async def test_update_api_key_disable(db_path):
    created = await create_api_key(db_path, name="toggle-key")
    await update_api_key(db_path, created["id"], enabled=False)
    fetched = await get_api_key(db_path, created["id"])
    assert fetched["enabled"] == 0

    # Re-enable
    await update_api_key(db_path, created["id"], enabled=True)
    fetched = await get_api_key(db_path, created["id"])
    assert fetched["enabled"] == 1


@pytest.mark.asyncio
async def test_mask_key():
    masked = mask_key("550e8400-e29b-41d4-a716-446655440000")
    assert masked == "550e****0000"
    assert len(masked) == 12
