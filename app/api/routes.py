"""Endpoints REST. La API solo valida, encola y consulta Redis: el grafo corre en el worker, nunca en el request.

    POST /tasks                      -> encola un turno (conversación nueva o existente) y devuelve job_id (202)
    GET  /tasks/{job_id}             -> estado del job (PENDING/RUNNING/WAITING_APPROVAL/DONE/FAILED)
    POST /tasks/{job_id}/approve     -> decisión humana para las tareas críticas
    GET  /conversations/{id}         -> turnos de una conversación
    GET  /health                     -> ping a Redis
"""
from fastapi import APIRouter, HTTPException, Request, status

from app.api.schemas import (
    ApprovalAccepted,
    ApprovalRequest,
    ConversationView,
    ErrorResponse,
    HealthResponse,
    TaskAccepted,
    TaskCreate,
    TaskView,
)
from app.core.jobs import ConversationBusy, ConversationNotFound, JobStatus, JobStore
from app.graph.hitl import classify_task

router = APIRouter()

_NOT_FOUND = {404: {"model": ErrorResponse}}
_CONFLICT = {409: {"model": ErrorResponse}}


def _store(request: Request) -> JobStore:
    return request.app.state.store


def _links(job_id: str, conversation_id: str) -> dict:
    return {"self": f"/tasks/{job_id}", "approve": f"/tasks/{job_id}/approve",
            "conversation": f"/conversations/{conversation_id}"}


@router.post("/tasks", response_model=TaskAccepted, status_code=status.HTTP_202_ACCEPTED, tags=["tasks"],
             responses={**_NOT_FOUND, **_CONFLICT})
async def create_task(body: TaskCreate, request: Request) -> TaskAccepted:
    """Encola la tarea y responde enseguida (no espera al LLM). Con `conversation_id` es una repregunta."""
    pending_action = classify_task(body.query, body.send_report_to)
    try:
        job = await _store(request).create(body.query, pending_action, body.conversation_id)
    except ConversationNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No existe la conversación {exc}") from exc
    except ConversationBusy as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    return TaskAccepted(job_id=job["job_id"], conversation_id=job["conversation_id"], status=JobStatus.PENDING,
                        critical=pending_action is not None, links=_links(job["job_id"], job["conversation_id"]))


@router.get("/tasks/{job_id}", response_model=TaskView, tags=["tasks"], responses=_NOT_FOUND)
async def get_task(job_id: str, request: Request) -> TaskView:
    """Estado actual del job (una lectura de Redis: pensado para polling)."""
    job = await _store(request).get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No existe el job {job_id}")
    return TaskView(**job)


@router.post("/tasks/{job_id}/approve", response_model=ApprovalAccepted, status_code=status.HTTP_202_ACCEPTED,
             tags=["human-in-the-loop"], responses={**_NOT_FOUND, **_CONFLICT})
async def approve_task(job_id: str, body: ApprovalRequest, request: Request) -> ApprovalAccepted:
    """Aprueba (o rechaza) la acción crítica de un job en WAITING_APPROVAL y encola su reanudación."""
    store = _store(request)
    job = await store.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No existe el job {job_id}")
    if not await store.request_resume(job_id, body.model_dump()):
        raise HTTPException(status.HTTP_409_CONFLICT,
                            f"El job {job_id} no está esperando aprobación (estado actual: {job['status']})")
    return ApprovalAccepted(job_id=job_id, status=JobStatus.PENDING, approval=body)


@router.get("/conversations/{conversation_id}", response_model=ConversationView, tags=["conversations"],
            responses=_NOT_FOUND)
async def get_conversation(conversation_id: str, request: Request) -> ConversationView:
    """Turnos de la conversación en orden, con su respuesta (el historial persistido en Redis)."""
    turns = await _store(request).conversation(conversation_id)
    if turns is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No existe la conversación {conversation_id}")
    return ConversationView(conversation_id=conversation_id, turns=[TaskView(**job) for job in turns])


@router.get("/health", response_model=HealthResponse, tags=["ops"], responses={503: {"model": ErrorResponse}})
async def health(request: Request) -> HealthResponse:
    try:
        await _store(request).redis.ping()
    except Exception as exc:  # noqa: BLE001 - el health check informa, no explota
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"Redis no responde: {exc}") from exc
    worker = "in-process" if request.app.state.runtime.worker else "external"
    return HealthResponse(status="ok", redis="ok", worker=worker)
