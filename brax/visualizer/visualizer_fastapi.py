"""FastAPI version of official brax's visualizer script."""

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import jax
from jax import numpy as jp
import mujoco

from brax.base import Actuator
from brax.generalized import pipeline as generalized
from brax.positional import pipeline as positional
from brax.spring import pipeline as spring
from brax.io import html, mjcf
from brax.io.mjcf import load_mjmodel
from etils import epath

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class _MujocoPipeline:
    @staticmethod
    def init(*args, **kwargs):
        return generalized.init(*args, **kwargs)

    @staticmethod
    def step(*args, **kwargs):
        return generalized.init(*args, **kwargs)

@app.get("/")
def index():
    return {"success": True}

@app.get("/favicon.ico")
def favicon():
    root_path = Path(__file__).parent
    favicon_path = root_path / "favicon.ico"
    if not favicon_path.exists():
        raise HTTPException(status_code=404, detail="favicon.ico not found")
    return FileResponse(str(favicon_path), media_type="image/x-icon")

@app.get("/js/{file_path:path}")
def serve_js(file_path: str):
    root_path = Path(__file__).parent
    js_file = root_path / "js" / file_path
    if not js_file.exists():
        raise HTTPException(status_code=404, detail="JS file not found")
    content = js_file.read_text()
    headers = {"Access-Control-Allow-Origin": "*"}
    return Response(content, media_type="text/javascript", headers=headers)

@app.get("/play/{file_path:path}", response_class=HTMLResponse)
def play_trajectory(file_path: str):
    try:
        system = epath.Path(file_path).read_text()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")
    rendered = html.render_from_json(
        system, height="100vh", colab=False, base_url="/js/viewer.js"
    )
    return HTMLResponse(content=rendered)

@app.get("/sim/{file_path:path}", response_class=HTMLResponse)
def simulate(
    file_path: str,
    pipeline: str = Query("generalized"),
    steps: int = Query(200),
    act: str = Query("sin"),
    solver_iterations: int = Query(10),
    add_act: bool = Query(False)
):
    try:
        sys = mjcf.load(file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading file: {str(e)}")

    pipeline_mapping = {
        "generalized": generalized,
        "positional": positional,
        "spring": spring,
        "mujoco": _MujocoPipeline,
    }
    if pipeline not in pipeline_mapping:
        raise HTTPException(status_code=400, detail="Invalid pipeline type")
    selected_pipeline = pipeline_mapping[pipeline]

    if solver_iterations > 0:
        sys = sys.replace(solver_iterations=solver_iterations)

    if add_act and not sys.actuator_types:
        actuator_types = "".join(["m" for t in sys.link_types if t in "123"])
        actuator_link_id = [i for i, t in enumerate(sys.link_types) if t in "123"]
        actuator_qid = [int(i) for i in sys.q_idx("123")]
        actuator_qdid = [int(i) for i in sys.qd_idx("123")]
        actuator = Actuator(
            ctrl_range=jp.tile(jp.array([-1.0, 1.0]), (len(actuator_types), 2)),
            gear=25 * jp.ones(len(actuator_types)),
        )
        sys = sys.replace(
            actuator=actuator,
            actuator_types=actuator_types,
            actuator_link_id=actuator_link_id,
            actuator_qid=actuator_qid,
            actuator_qdid=actuator_qdid,
        )

    if pipeline == "mujoco":
        mj_model = load_mjmodel(file_path)
        mj_data = mujoco.MjData(mj_model)
        init_fn = jax.jit(selected_pipeline.init)

        def step_fn(sys, prev_state, act_val):
            mj_data.ctrl = act_val
            mujoco.mj_step(mj_model, mj_data)
            state = init_fn(sys, mj_data.qpos, jp.zeros(sys.qd_size()))
            return state
    else:
        init_fn = jax.jit(selected_pipeline.init)
        step_fn = jax.jit(selected_pipeline.step)

    state = init_fn(sys, sys.init_q, jp.zeros(sys.qd_size()))
    states = [state]
    for i in range(steps):
        if act == "sin":
            act_val = 0.5 * jp.sin(jp.ones(sys.act_size()) * 5 * i * sys.opt.timestep)
        elif act == "zero":
            act_val = jp.zeros(sys.act_size())
        elif act == "zero_p":
            q = states[-1].q[sys.q_idx("123")]
            act_val = -q
        else:
            raise HTTPException(status_code=400, detail=f"Unknown act function: {act}")
        state = step_fn(sys, states[-1], act_val)
        states.append(state)

    rendered = html.render(
        sys, states, height="100vh", colab=False, base_url="/js/viewer.js"
    )
    return HTMLResponse(content=rendered)

@app.get("/live/{file_path:path}", response_class=HTMLResponse)
def live_simulate(
    file_path: str,
    pipeline: str = Query("generalized"),
    steps: int = Query(200),
    act: str = Query("sin"),
    solver_iterations: int = Query(10),
    add_act: bool = Query(False)
):
    try:
        sys = mjcf.load(file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading file: {str(e)}")

    pipeline_mapping = {
        "generalized": generalized,
        "positional": positional,
        "spring": spring,
        "mujoco": _MujocoPipeline,
    }
    if pipeline not in pipeline_mapping:
        raise HTTPException(status_code=400, detail="Invalid pipeline type")
    selected_pipeline = pipeline_mapping[pipeline]

    if solver_iterations > 0:
        sys = sys.replace(solver_iterations=solver_iterations)

    if add_act and not sys.actuator_types:
        actuator_types = "".join(["m" for t in sys.link_types if t in "123"])
        actuator_link_id = [i for i, t in enumerate(sys.link_types) if t in "123"]
        actuator_qid = [int(i) for i in sys.q_idx("123")]
        actuator_qdid = [int(i) for i in sys.qd_idx("123")]
        actuator = Actuator(
            ctrl_range=jp.tile(jp.array([-1.0, 1.0]), (len(actuator_types), 2)),
            gear=25 * jp.ones(len(actuator_types)),
        )
        sys = sys.replace(
            actuator=actuator,
            actuator_types=actuator_types,
            actuator_link_id=actuator_link_id,
            actuator_qid=actuator_qid,
            actuator_qdid=actuator_qdid,
        )

    if pipeline == "mujoco":
        mj_model = load_mjmodel(file_path)
        mj_data = mujoco.MjData(mj_model)
        init_fn = jax.jit(selected_pipeline.init)

        def step_fn(sys, prev_state, act_val):
            mj_data.ctrl = act_val
            mujoco.mj_step(mj_model, mj_data)
            state = init_fn(sys, mj_data.qpos, jp.zeros(sys.qd_size()))
            return state
    else:
        init_fn = jax.jit(selected_pipeline.init)
        step_fn = jax.jit(selected_pipeline.step)

    # state = init_fn(sys, sys.init_q, jp.zeros(sys.qd_size()))
    # states = [state]
    # for i in range(steps):
    #     if act == "sin":
    #         act_val = 0.5 * jp.sin(jp.ones(sys.act_size()) * 5 * i * sys.opt.timestep)
    #     elif act == "zero":
    #         act_val = jp.zeros(sys.act_size())
    #     elif act == "zero_p":
    #         q = states[-1].q[sys.q_idx("123")]
    #         act_val = -q
    #     else:
    #         raise HTTPException(status_code=400, detail=f"Unknown act function: {act}")
    #     state = step_fn(sys, states[-1], act_val)
    #     states.append(state)

    # rendered = html.render(
    #     sys, states, height="100vh", colab=False, base_url="/js/viewer.js"
    # )
    # return HTMLResponse(content=rendered)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(__name__+":app", host="localhost", port=8080, reload=True)
