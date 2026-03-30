"""
StreamLens — WebSocket Real-Time Feed Updates
Pushes updated rankings to connected clients when model updates.
"""
from __future__ import annotations
import json, asyncio, time


class ConnectionManager:
    def __init__(self):
        self._connections: dict[str, list] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket, user_id: str) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.setdefault(user_id, []).append(websocket)

    async def disconnect(self, websocket, user_id: str) -> None:
        async with self._lock:
            if user_id in self._connections:
                try: self._connections[user_id].remove(websocket)
                except ValueError: pass
                if not self._connections[user_id]:
                    del self._connections[user_id]

    async def send_to_user(self, user_id: str, message: dict) -> int:
        payload = json.dumps(message)
        sent, dead = 0, []
        for ws in self._connections.get(user_id, []):
            try: await ws.send_text(payload); sent += 1
            except Exception: dead.append(ws)
        for ws in dead: await self.disconnect(ws, user_id)
        return sent

    async def broadcast(self, message: dict) -> int:
        payload = json.dumps(message)
        sent = 0
        for conns in list(self._connections.values()):
            for ws in conns:
                try: await ws.send_text(payload); sent += 1
                except Exception: pass
        return sent

    @property
    def active_users(self): return len(self._connections)

    @property
    def active_connections(self): return sum(len(v) for v in self._connections.values())


def make_feed_update(user_id, ranked_items, trigger="model_update"):
    return {"type": "feed_update", "user_id": user_id, "trigger": trigger,
            "items": ranked_items[:10], "timestamp": time.time(),
            "model_version": "ltr_e5base_v2"}

def make_interaction_ack(event_id, doc_id, event_type):
    return {"type": "interaction_ack", "event_id": event_id,
            "doc_id": doc_id, "event_type": event_type, "timestamp": time.time()}

def make_model_retrain_notification(new_ndcg, old_ndcg):
    return {"type": "model_retrained", "old_ndcg": old_ndcg, "new_ndcg": new_ndcg,
            "improvement_pct": round(100*(new_ndcg-old_ndcg)/max(old_ndcg,0.001), 2),
            "timestamp": time.time()}


_MANAGER = None
def get_manager() -> ConnectionManager:
    global _MANAGER
    if _MANAGER is None: _MANAGER = ConnectionManager()
    return _MANAGER
