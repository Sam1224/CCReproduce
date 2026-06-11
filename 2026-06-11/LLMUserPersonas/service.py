import asyncio
from dataclasses import dataclass

from persona import Persona


@dataclass
class PersonaRequest:
    user_id: int
    cluster_video_ids: list[int]
    history_video_ids: set[int]


@dataclass
class PersonaResponse:
    user_id: int
    personas: list[Persona]


class AsyncPersonaService:
    def __init__(self, generator, videos):
        self.generator = generator
        self.videos = videos
        self.queue: asyncio.Queue[tuple[PersonaRequest, asyncio.Future]] = asyncio.Queue()
        self._task = None

    async def start(self):
        if self._task is None:
            self._task = asyncio.create_task(self._worker())

    async def stop(self):
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def submit(self, req: PersonaRequest) -> PersonaResponse:
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        await self.queue.put((req, fut))
        return await fut

    async def _worker(self):
        while True:
            req, fut = await self.queue.get()
            try:
                personas = self.generator.generate(self.videos, req.cluster_video_ids, req.history_video_ids)
                fut.set_result(PersonaResponse(user_id=req.user_id, personas=personas))
            except Exception as e:  # noqa: BLE001
                fut.set_exception(e)
            finally:
                self.queue.task_done()
