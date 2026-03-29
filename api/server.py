from fastapi import FastAPI

from api.routes import router


def create_app() -> FastAPI:
	app = FastAPI(title="Pharmacy AI Verify API", version="1.0.0")
	app.include_router(router)
	return app


app = create_app()
