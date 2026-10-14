# Entrega final: Sistema Intelligence de grado de producción

## Lo esencial en 3 líneas

- **Qué entregás:** un repositorio de GitHub con un sistema de IA completo que integra RAG + agentes (con Supervisor) + API profesional + trazabilidad.
- **De dónde sale:** ensamblás y refactorizás tus pre-entregas de los módulos anteriores (todo async, con Checkpointer de LangGraph y validación con Pydantic).
- **Cómo se evalúa:** con la rúbrica de la entrega final (más abajo en esta unidad).

## El Camino Recorrido

Para llegar aquí, has superado hitos técnicos fundamentales que ahora se ensamblan:

- **Cimientos Asíncronos & Abstracción:** Implementaste clientes robustos que manejan streaming y latencia.
- **RAG Avanzado:** Diseñaste estrategias de recuperación híbrida y gestión de metadatos para reducir alucinaciones.
- **Inteligencia Colectiva:** Creaste una arquitectura de agentes donde un Supervisor delega tareas de forma autónoma.
- **Robustez & Despliegue:** Preparaste el entorno para producción con observabilidad (Arize Phoenix) y escalabilidad (Redis/FastAPI).

## Descripción del Entregable

Debes entregar un repositorio de GitHub que contenga un sistema de IA completo (*"The Intelligence System"*). Este sistema debe permitir a un usuario final (o proceso automatizado) resolver una tarea compleja que requiera:

1. Acceso a conocimiento específico (vía **RAG**).
2. Razonamiento multi-paso (vía **Agentes**).
3. Interfaz de **API** profesional y documentada.
4. **Trazabilidad** completa de lo que ocurre "bajo el capó".

## Errores Comunes a Evitar

- **Hard-coding:** Evita poner IDs de índices de Pinecone o modelos directamente en el código; usa variables de entorno.
- **Sincronicidad:** No mezcles llamadas síncronas (`requests`) dentro de funciones `async def`; usa `httpx` o los SDKs asíncronos oficiales.
- **Falta de Trazas:** Entregar un sistema que "funciona" pero cuya ejecución es una caja negra (sin integración de logs o traces).

> **Alcance del despliegue:** este entregable se evalúa con despliegue local (Docker Compose o `run.sh`). No se pide CI/CD (GitHub Actions, pipelines de integración continua) ni infraestructura en la nube: no suma ni resta en la rúbrica.

## Cierre del Programa

Este sistema es tu carta de presentación como AI Architect. Has construido una pieza de ingeniería capaz de razonar, recordar y escalar. Tras esta entrega, estarás listo para enfrentar retos de implementación de IA en entornos corporativos reales. ¡Es hora de consolidar tu visión! Estamos ansiosos por ver tu arquitectura en funcionamiento.

---

> 💡 ¡Ya está disponible el material! Hacé clic en el enlace de abajo para repasar la presentación de la clase de hoy y refrescar los conceptos clave.
>
> 🔗 [Ver presentación aquí](<Proyecto final _ AI Engineering (V2).pptx.pdf>)

---

## Entregable

Repositorio de GitHub con el código del sistema completo (**Python 3.12+**), archivos de configuración, documentación técnica y trazas de observabilidad configuradas.

- [ ] **Integración de Código:** Mueve la lógica de tus pre-entregas de los Módulos 6 y 7 a una estructura de proyecto unificada.
- [ ] **Refactorización:** Asegúrate de que toda la comunicación con LLMs y bases de datos sea asíncrona (`async`/`await`).
- [ ] **Persistencia:** Implementa un Checkpointer en LangGraph para que las conversaciones no se pierdan al reiniciar el servidor.
- [ ] **Validación:** Usa Pydantic para validar todas las entradas y salidas de la API y de las herramientas de los agentes.
- [ ] **Observabilidad:** Realiza al menos 5 pruebas del sistema y asegúrate de que las trazas se visualicen correctamente en Arize Phoenix o LangSmith.
- [ ] **README:** Escribe la documentación incluyendo un diagrama de flujo del grafo de agentes.
- [ ] **Despliegue Local:** Asegúrate de que el proyecto pueda levantarse con una sola instrucción (ej. Docker Compose o un script `run.sh`).
