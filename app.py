from flask import Flask, request, jsonify
import pymunk
from pymunk.vec2d import Vec2d
import numpy as np
from typing import List, Dict, Optional
import threading
import time
import math
import uuid
import random

app = Flask(__name__, static_folder="static", static_url_path="")

# ------------------------------
# 物理世界 / 常量
# ------------------------------
space = pymunk.Space()
space.gravity = (0, 0)
space.damping = 1.00
space.iterations = 30
G = 6.67e-11

node_radius = 1.0
engi_radius = 1.0
node_mass = 1000.0
node_max_force = 1000.0
engi_mass = 1000.0
engi_max_force = 200.0
core_max_force = 1e9
engi_force_k = 50.0

plank_radius = 1.0
mass_per_length = 50.0

rot_stiffness = 8000.0
rot_damping   = 2500.0

TURN_ALPHA = 2.8
MAX_MANUAL_TORQUE = 2e7

turn_angle_speed = np.pi*(25/180)

att_kp = 8
att_kd = 6
MAX_ATT_TORQUE = 6e6

Fixed_fps = 60

# ------------------------------
# 全局锁
# ------------------------------
state_lock = threading.Lock()

# ------------------------------
# 实体类
# ------------------------------
class Planet:
    def __init__(self, mass: float, R: float, name: str = "planet"):
        self.name = name
        self.radius = R
        moment = pymunk.moment_for_circle(mass=mass, inner_radius=0, outer_radius=R)
        self.body = pymunk.Body(mass=mass, moment=moment)
        self.body.position = (0, 0)

        shape = pymunk.Circle(self.body, R)
        shape.friction = 0.5
        shape.elasticity = 0.0
        space.add(self.body, shape)
        self.shape = shape

Planets: List[Planet] = []


class Node:
    def __init__(self, pos: Vec2d = Vec2d(0, 0), max_force=node_max_force, group=0):
        self.radius = node_radius
        moment = pymunk.moment_for_circle(mass=node_mass, inner_radius=0, outer_radius=self.radius)
        self.body = pymunk.Body(mass=node_mass, moment=moment)
        self.body.position = pos

        shape = pymunk.Circle(self.body, self.radius)
        shape.friction = 0.5
        shape.elasticity = 0.0
        shape.filter = pymunk.ShapeFilter(group=group)
        space.add(self.body, shape)

        self.shape = shape
        self.max_force = max_force


class Engine:
    def __init__(
        self,
        pos: Vec2d = Vec2d(0, 0),
        max_force=engi_max_force,
        rotation: float = 0.0,
        force: float = 0.0,
        radius: float = engi_radius,
        mass: float = engi_mass,
        group=0,
    ):
        self.radius = radius
        moment = pymunk.moment_for_circle(mass=mass, inner_radius=0, outer_radius=self.radius)
        self.body = pymunk.Body(mass=mass, moment=moment)
        self.body.position = pos
        self.body.angle = rotation

        shape = pymunk.Circle(self.body, self.radius)
        shape.friction = 0.5
        shape.elasticity = 0.0
        shape.filter = pymunk.ShapeFilter(group=group)
        space.add(self.body, shape)

        self.shape = shape
        self.max_force = max_force
        self.force = force


class Plank:
    def __init__(self, obj_a, obj_b, thickness: float = plank_radius):
        self.obj_a = obj_a
        self.obj_b = obj_b

        body_a = obj_a.body
        body_b = obj_b.body

        p1 = body_a.position
        p2 = body_b.position
        v = p2 - p1
        length = v.length
        if length < 1e-6:
            length = 1e-6
        mid = (p1 + p2) / 2.0
        angle = v.angle

        self.length = length
        self.radius = thickness / 2.0

        mass = mass_per_length * length
        half_len = length / 2.0
        a_local = Vec2d(-half_len, 0.0)
        b_local = Vec2d( half_len, 0.0)
        moment = pymunk.moment_for_segment(mass, a_local, b_local, self.radius)

        self.body = pymunk.Body(mass, moment)
        self.body.position = mid
        self.body.angle = angle

        self.shape = pymunk.Segment(self.body, a_local, b_local, self.radius)
        self.shape.friction = 0.5
        self.shape.elasticity = 0.0
        group = obj_a.shape.filter.group
        self.shape.filter = pymunk.ShapeFilter(group=group)

        space.add(self.body, self.shape)

        joints = []

        world_anchor1 = p1
        world_anchor2 = p2

        anchor_plank_a = self.body.world_to_local(world_anchor1)
        anchor_node_a  = body_a.world_to_local(world_anchor1)

        anchor_plank_b = self.body.world_to_local(world_anchor2)
        anchor_node_b  = body_b.world_to_local(world_anchor2)

        self.pivot_a = pymunk.PivotJoint(self.body, body_a, anchor_plank_a, anchor_node_a)
        self.pivot_b = pymunk.PivotJoint(self.body, body_b, anchor_plank_b, anchor_node_b)
        joints.extend([self.pivot_a, self.pivot_b])

        rest_angle_a = self.body.angle - body_a.angle
        rest_angle_b = self.body.angle - body_b.angle

        self.spring_a = pymunk.DampedRotarySpring(
            self.body, body_a, rest_angle_a, rot_stiffness, rot_damping
        )
        self.spring_b = pymunk.DampedRotarySpring(
            self.body, body_b, rest_angle_b, rot_stiffness, rot_damping
        )
        joints.extend([self.spring_a, self.spring_b])

        space.add(*joints)


class Ship:
    uid_counter = 0

    def __init__(self, pos=(0, 0), owner_uid: str = ""):
        Ship.uid_counter += 1
        self.id = Ship.uid_counter
        self.owner_uid = owner_uid

        self.nodes: List[Node] = [Node(pos=Vec2d(*pos), max_force=core_max_force, group=self.id)]
        self.engis: List[Engine] = []
        self.planks: List[Plank] = []

        self.on_fire = False
        self.on_left = False
        self.on_right = False

        self.stabilize = False
        self.stable_angle = 0.0

        # 多人关键：每艘船自己的“上一次 ship_angle”，别用全局变量
        self.last_ship_angle = 0.0

    @property
    def core(self) -> Node:
        return self.nodes[0]


Ships: List[Ship] = []
ShipsByUid: Dict[str, Ship] = {}

# ------------------------------
# 工具函数
# ------------------------------
def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi

def compute_ship_angle(ship: Ship) -> float:
    core_pos = ship.core.body.position
    if ship.engis:
        center = Vec2d(0, 0)
        for e in ship.engis:
            center += e.body.position
        center /= len(ship.engis)

        dir_vec = core_pos - center
        if dir_vec.length > 1e-6:
            return dir_vec.angle
    return 0.0

def ship_bodies(s: Ship):
    return [n.body for n in s.nodes] + [e.body for e in s.engis] + [pl.body for pl in s.planks]

def compute_ship_com_and_inertia(s: Ship):
    bodies = ship_bodies(s)
    total_m = sum(b.mass for b in bodies)
    if total_m <= 1e-9:
        return Vec2d(0, 0), 0.0

    com = Vec2d(0, 0)
    for b in bodies:
        com += b.position * b.mass
    com /= total_m

    I_total = 0.0
    for b in bodies:
        r = b.position - com
        I_total += b.moment + b.mass * r.length_squared
    return com, I_total

# ------------------------------
# 万有引力 & 控制
# ------------------------------
def Planet2Planet(dt: float):
    n = len(Planets)
    for i in range(n):
        for j in range(i):
            pi = Planets[i]
            pj = Planets[j]
            tow = pj.body.position - pi.body.position
            dist = tow.length
            if dist <= 1e-6:
                continue
            dir_vec = tow / dist
            mag = G * pi.body.mass * pj.body.mass / (dist**2)
            force = dir_vec * mag
            pi.body.apply_force_at_world_point(force, pi.body.position)
            pj.body.apply_force_at_world_point(-force, pj.body.position)

def Planet2Ship(dt: float):
    for planet in Planets:
        R = planet.radius
        M = planet.body.mass

        for s in Ships:
            bodies = [n.body for n in s.nodes] + [e.body for e in s.engis] + [pl.body for pl in s.planks]

            for b in bodies:
                px, py = b.position
                if not (math.isfinite(px) and math.isfinite(py)):
                    continue

                r_vec = planet.body.position - b.position
                dist = r_vec.length
                if dist <= 1e-6:
                    continue

                eff_r = max(dist, R)
                dir_vec = r_vec / eff_r
                g_mag = G * M / (eff_r**2)

                force = dir_vec * (b.mass * g_mag)
                b.apply_force_at_world_point(force, b.position)

def Control_Ship(dt: float):
    for s in Ships:
        core_body = s.core.body
        ship_ang = compute_ship_angle(s)

        # 1) 手动旋转（未按 Shift）
        if not s.stabilize:
            _, I_total = compute_ship_com_and_inertia(s)
            base_tau = I_total * TURN_ALPHA
            if base_tau > MAX_MANUAL_TORQUE:
                base_tau = MAX_MANUAL_TORQUE

            if s.on_left:
                core_body.torque += base_tau
            if s.on_right:
                core_body.torque -= base_tau
        else:
            # Shift 模式下：左右键改变 stable_angle
            if s.on_left:
                s.stable_angle += turn_angle_speed * dt
            if s.on_right:
                s.stable_angle -= turn_angle_speed * dt

        # 2) 姿态稳定：锁到 stable_angle
        if s.stabilize:
            current_angle = ship_ang
            angle_error = wrap_angle(s.stable_angle - current_angle)

            # 用 wrap 后的 delta 算角速度，避免跨 ±pi 时爆炸
            delta = wrap_angle(current_angle - s.last_ship_angle)
            ang_vel = delta / dt if dt > 1e-9 else 0.0

            _, I_total = compute_ship_com_and_inertia(s)
            torque = I_total * (att_kp * angle_error - att_kd * ang_vel)

            if torque > MAX_ATT_TORQUE:
                torque = MAX_ATT_TORQUE
            elif torque < -MAX_ATT_TORQUE:
                torque = -MAX_ATT_TORQUE

            core_body.torque += torque
            s.last_ship_angle = current_angle

        # 3) 引擎推力：沿 ship_angle
        if s.on_fire and s.engis:
            thrust_dir = Vec2d(np.cos(ship_ang), np.sin(ship_ang))
            for e in s.engis:
                force = thrust_dir * (e.body.mass * engi_force_k)
                e.body.apply_force_at_world_point(force, e.body.position)

# ------------------------------
# 物理更新
# ------------------------------
def FixedUpdate(dt: float):
    Planet2Planet(dt)
    Planet2Ship(dt)
    Control_Ship(dt)
    space.step(dt)

def Physics_loop():
    T = 1.0 / Fixed_fps
    while True:
        before = time.perf_counter()
        with state_lock:
            FixedUpdate(T)
        now = time.perf_counter() - before
        if T > now:
            time.sleep(T - now)

# ------------------------------
# 世界初始化：一个星球（不再默认生成一艘船）
# ------------------------------
def init_world():
    planet_R = 20000.0
    planet_g = 9.8
    planet_M = planet_g * planet_R**2 / G
    earth = Planet(mass=planet_M, R=planet_R, name="home")
    Planets.append(earth)

    E2R_Moon=5e4
    R_Moon=5e3
    g_Moon=1.625

    Moon_M=g_Moon*R_Moon**2/G
    moon = Planet(mass=Moon_M, R=R_Moon, name="moon")
    Planets.append(moon)
    omega=np.sqrt(G*(planet_M+Moon_M)/(E2R_Moon**3))
    v=omega*E2R_Moon
    moon.body.position=(E2R_Moon,0)
    moon.body.velocity=(0,v)


def create_ship_for_player(owner_uid: str) -> Ship:
    # 在星球表面随机角度生成
    planet = Planets[0]
    R = planet.radius
    start_alt = 200.0

    theta = random.random() * 2 * math.pi
    u = Vec2d(math.cos(theta), math.sin(theta))  # 径向单位向量

    core_pos = u * (R + start_alt)
    ship = Ship(pos=(core_pos.x, core_pos.y), owner_uid=owner_uid)

    core = ship.core
    # 引擎放在“靠近星球”方向（core 往 -u）
    eng_pos = core.body.position - u * 10.0

    # 让引擎角度=飞船朝向（指向 +u），这样前端画出来推力方向好看
    eng = Engine(pos=eng_pos, rotation=u.angle, group=ship.id)
    ship.engis.append(eng)
    ship.planks.append(Plank(core, eng))

    ship.stable_angle = compute_ship_angle(ship)
    ship.last_ship_angle = ship.stable_angle
    return ship

def get_ship_by_uid(uid: str) -> Optional[Ship]:
    return ShipsByUid.get(uid)

# ------------------------------
# HTTP 路由
# ------------------------------
@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/api/join", methods=["POST"])
def api_join():
    """
    每次页面加载调用：分配一个新的 uid，并生成一艘新船
    返回：{ ok, uid, ship_id }
    """
    uid = uuid.uuid4().hex

    with state_lock:
        ship = create_ship_for_player(uid)
        Ships.append(ship)
        ShipsByUid[uid] = ship

    return jsonify({"ok": True, "uid": uid, "ship_id": int(ship.id)})

@app.route("/api/state", methods=["GET"])
def api_state():
    with state_lock:
        planets_json = []
        for p in Planets:
            planets_json.append({
                "name": p.name,
                "x": float(p.body.position.x),
                "y": float(p.body.position.y),
                "radius": float(p.radius),
            })

        ships_json = []
        for s in Ships:
            nodes_json = []
            for idx, n in enumerate(s.nodes):
                nodes_json.append({
                    "id": idx,
                    "x": float(n.body.position.x),
                    "y": float(n.body.position.y),
                    "vx": float(n.body.velocity.x),
                    "vy": float(n.body.velocity.y),
                    "ang_vel": float(n.body.angular_velocity),
                    "angle": float(n.body.angle),
                    "radius": float(n.radius),
                })

            engis_json = []
            for e in s.engis:
                engis_json.append({
                    "x": float(e.body.position.x),
                    "y": float(e.body.position.y),
                    "vx": float(e.body.velocity.x),
                    "vy": float(e.body.velocity.y),
                    "ang_vel": float(e.body.angular_velocity),
                    "angle": float(e.body.angle),
                    "radius": float(e.radius),
                })

            planks_json = []
            for pl in s.planks:
                planks_json.append({
                    "x": float(pl.body.position.x),
                    "y": float(pl.body.position.y),
                    "vx": float(pl.body.velocity.x),
                    "vy": float(pl.body.velocity.y),
                    "ang_vel": float(pl.body.angular_velocity),
                    "angle": float(pl.body.angle),
                    "radius": float(pl.radius),
                    "length": float(pl.length),
                })

            ship_angle = float(compute_ship_angle(s))
            ships_json.append({
                "id": s.id,
                "on_fire": s.on_fire,
                "on_left": s.on_left,
                "on_right": s.on_right,
                "stabilize": s.stabilize,
                "ship_angle": ship_angle,
                "nodes": nodes_json,
                "engines": engis_json,
                "planks": planks_json,
            })

    return jsonify({"planets": planets_json, "ships": ships_json})

@app.route("/api/control", methods=["POST"])
def api_control():
    """
    { uid, fire, left, right, stabilize }
    只控制自己的船
    """
    data = request.get_json(force=True) or {}
    uid = str(data.get("uid", "")).strip()
    if not uid:
        return jsonify({"ok": False, "error": "uid required"}), 400

    fire = bool(data.get("fire", False))
    left = bool(data.get("left", False))
    right = bool(data.get("right", False))
    stabilize = bool(data.get("stabilize", False))
    with state_lock:
        s = get_ship_by_uid(uid)
        if s is None:
            return jsonify({"ok": False, "error": "ship not found for uid"}), 404

        s.on_fire = fire
        s.on_left = left
        s.on_right = right

        # Shift 刚按下：锁定当前 ship_angle
        if stabilize and not s.stabilize:
            s.stable_angle = compute_ship_angle(s)
            s.last_ship_angle = s.stable_angle

        s.stabilize = stabilize
    return jsonify({"ok": True})

# ------------------------------
# 建造：木板
# ------------------------------
@app.route("/api/build_plank", methods=["POST"])
def api_build_plank():
    """
    要求 uid，只能改自己的船
    """
    data = request.get_json(force=True) or {}
    uid = str(data.get("uid", "")).strip()
    if not uid:
        return jsonify({"ok": False, "error": "uid required"}), 400

    ship_id = int(data.get("ship_id", 0))

    def get_obj(ship: Ship, ref: dict):
        t = (ref.get("type") or "node").lower()
        idx = int(ref.get("index", -1))
        if t == "node":
            if idx < 0 or idx >= len(ship.nodes):
                return None, "node index invalid"
            return ship.nodes[idx], None
        elif t == "engine":
            if idx < 0 or idx >= len(ship.engis):
                return None, "engine index invalid"
            return ship.engis[idx], None
        return None, "unknown ref type"

    with state_lock:
        ship = get_ship_by_uid(uid)
        if ship is None or ship.id != ship_id:
            return jsonify({"ok": False, "error": "permission denied"}), 403

        from_ref = data.get("from", None)
        if from_ref is None:
            from_ref = {"type": "node", "index": int(data.get("from_node", -1))}

        obj_a, err = get_obj(ship, from_ref)
        if err:
            return jsonify({"ok": False, "error": f"from: {err}"}), 400

        new_node_json = None
        to_ref = data.get("to", None)
        to_node_compat = int(data.get("to_node", 999999))

        if to_ref is not None:
            obj_b, err = get_obj(ship, to_ref)
            if err:
                return jsonify({"ok": False, "error": f"to: {err}"}), 400
        else:
            if to_node_compat == -1:
                new_node_pos = data.get("new_node_pos")
                if not new_node_pos or "x" not in new_node_pos or "y" not in new_node_pos:
                    return jsonify({"ok": False, "error": "new_node_pos required when to_node=-1"}), 400

                pos = Vec2d(float(new_node_pos["x"]), float(new_node_pos["y"]))
                new_node = Node(pos=pos, max_force=node_max_force, group=ship.id)
                new_node.body.velocity = obj_a.body.velocity

                ship.nodes.append(new_node)
                obj_b = new_node

                new_idx = len(ship.nodes) - 1
                new_node_json = {
                    "id": new_idx,
                    "x": float(new_node.body.position.x),
                    "y": float(new_node.body.position.y),
                    "angle": float(new_node.body.angle),
                    "radius": float(new_node.radius),
                }
            else:
                obj_b, err = get_obj(ship, {"type": "node", "index": int(data.get("to_node", -1))})
                if err:
                    return jsonify({"ok": False, "error": f"to: {err}"}), 400

        if obj_a is obj_b:
            return jsonify({"ok": False, "error": "cannot connect to itself"}), 400

        plank = Plank(obj_a, obj_b)
        plank.body.velocity = obj_a.body.velocity
        ship.planks.append(plank)

        mid = plank.body.position
        plank_json = {
            "x": float(mid.x),
            "y": float(mid.y),
            "angle": float(plank.body.angle),
            "radius": float(plank.radius),
            "length": float(plank.length),
        }

    return jsonify({"ok": True, "plank": plank_json, "new_node": new_node_json})

# ------------------------------
# 建造：引擎
# ------------------------------
@app.route("/api/build_engine", methods=["POST"])
def api_build_engine():
    """
    要求 uid，只能改自己的船
    JSON:
    { uid, ship_id, from_node, target:{x,y} }
    """
    data = request.get_json(force=True) or {}
    uid = str(data.get("uid", "")).strip()
    if not uid:
        return jsonify({"ok": False, "error": "uid required"}), 400

    ship_id = int(data.get("ship_id", 0))
    from_idx = int(data.get("from_node", -1))
    target = data.get("target") or {}
    tx = float(target.get("x", 0.0))
    ty = float(target.get("y", 0.0))

    with state_lock:
        ship = get_ship_by_uid(uid)
        if ship is None or ship.id != ship_id:
            return jsonify({"ok": False, "error": "permission denied"}), 403

        if from_idx < 0 or from_idx >= len(ship.nodes):
            return jsonify({"ok": False, "error": "from_node index invalid"}), 400

        node = ship.nodes[from_idx]
        base_pos = node.body.position
        dir_vec = Vec2d(tx, ty) - base_pos
        length = dir_vec.length
        if length < 1e-3:
            return jsonify({"ok": False, "error": "drag too short"}), 400

        dir_unit = dir_vec / length

        ENGINE_MIN_R = 1.0
        ENGINE_MAX_R = 5.0
        ENGINE_LEN2R = 0.2

        radius = length * ENGINE_LEN2R
        radius = max(ENGINE_MIN_R, min(ENGINE_MAX_R, radius))
        mass = engi_mass * (radius / engi_radius) ** 2

        offset = node.radius + radius + 2.0
        eng_pos = base_pos + dir_unit * offset

        angle = dir_unit.angle + np.pi  # 保持你原来的“推力方向”定义

        eng = Engine(
            pos=eng_pos,
            rotation=angle,
            radius=radius,
            mass=mass,
            group=ship.id,
        )
        eng.body.velocity = node.body.velocity
        ship.engis.append(eng)

        plank = Plank(node, eng)
        plank.body.velocity = node.body.velocity
        ship.planks.append(plank)

        engine_json = {
            "x": float(eng.body.position.x),
            "y": float(eng.body.position.y),
            "angle": float(eng.body.angle),
            "radius": float(eng.radius),
        }

        mid = plank.body.position
        plank_json = {
            "x": float(mid.x),
            "y": float(mid.y),
            "angle": float(plank.body.angle),
            "radius": float(plank.radius),
            "length": float(plank.length),
        }

    return jsonify({"ok": True, "engine": engine_json, "plank": plank_json})

# ------------------------------
# 删除：node / engine / plank
# ------------------------------
@app.route("/api/delete", methods=["POST"])
def api_delete():
    """
    要求 uid，只能删自己的船
    { uid, ship_id, target:{type, index} }
    """
    data = request.get_json(force=True) or {}
    uid = str(data.get("uid", "")).strip()
    if not uid:
        return jsonify({"ok": False, "error": "uid required"}), 400

    ship_id = int(data.get("ship_id", 0))
    target = data.get("target") or {}
    ttype = str(target.get("type", "")).lower()
    idx = int(target.get("index", -1))

    def safe_remove(obj):
        if obj is None:
            return
        try:
            space.remove(obj)
        except Exception:
            pass

    def remove_plank(pl: Plank):
        safe_remove(pl.pivot_a)
        safe_remove(pl.pivot_b)
        safe_remove(pl.spring_a)
        safe_remove(pl.spring_b)
        safe_remove(pl.shape)
        safe_remove(pl.body)

    with state_lock:
        ship = get_ship_by_uid(uid)
        if ship is None or ship.id != ship_id:
            return jsonify({"ok": False, "error": "permission denied"}), 403

        if ttype == "plank":
            if idx < 0 or idx >= len(ship.planks):
                return jsonify({"ok": False, "error": "plank index invalid"}), 400
            pl = ship.planks[idx]
            remove_plank(pl)
            del ship.planks[idx]
            return jsonify({"ok": True})

        if ttype == "node":
            if idx <= 0:
                return jsonify({"ok": False, "error": "cannot delete core node"}), 400
            if idx >= len(ship.nodes):
                return jsonify({"ok": False, "error": "node index invalid"}), 400

            node = ship.nodes[idx]

            connected = [pl for pl in ship.planks if (pl.obj_a is node or pl.obj_b is node)]
            for pl in connected:
                remove_plank(pl)
                try:
                    ship.planks.remove(pl)
                except ValueError:
                    pass

            safe_remove(node.shape)
            safe_remove(node.body)
            del ship.nodes[idx]
            return jsonify({"ok": True})

        if ttype == "engine":
            if idx < 0 or idx >= len(ship.engis):
                return jsonify({"ok": False, "error": "engine index invalid"}), 400

            eng = ship.engis[idx]

            connected = [pl for pl in ship.planks if (pl.obj_a is eng or pl.obj_b is eng)]
            for pl in connected:
                remove_plank(pl)
                try:
                    ship.planks.remove(pl)
                except ValueError:
                    pass

            safe_remove(eng.shape)
            safe_remove(eng.body)
            del ship.engis[idx]
            return jsonify({"ok": True})

        return jsonify({"ok": False, "error": "unknown target type"}), 400

# ------------------------------
# 预测：只允许预测自己的船
# ------------------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    超轻量预测：只考虑“自己飞船的 core”和“行星引力”
    { uid, ship_id, seconds, dt, mode, stride }
    mode:
      - "ballistic" / "ballistic_fast": 默认使用轻量预测
      - "current": 你如果以后要加推力预测再说（这里先当 ballistic 处理）
    """
    data = request.get_json(force=True) or {}
    uid = str(data.get("uid", "")).strip()
    if not uid:
        return jsonify({"ok": False, "error": "uid required"}), 400

    req_ship_id = int(data.get("ship_id", 0))
    seconds = float(data.get("seconds", 20.0))
    dt = float(data.get("dt", 1.0 / 60.0))
    mode = str(data.get("mode", "ballistic")).lower()
    stride = int(data.get("stride", 2))
    if stride < 1:
        stride = 1

    # --- 在锁内：只抓“最少必需快照” ---
    with state_lock:
        ship = get_ship_by_uid(uid)
        if ship is None or ship.id != req_ship_id:
            return jsonify({"ok": False, "error": "permission denied"}), 403

        # core 的状态（只要它）
        core = ship.core.body
        core_pos = (float(core.position.x), float(core.position.y))
        core_vel = (float(core.velocity.x), float(core.velocity.y))

        # 行星快照（质量/半径/位置/速度）
        planets = []
        for p in Planets:
            planets.append({
                "name": p.name,
                "mass": float(p.body.mass),
                "radius": float(p.radius),
                "pos": [float(p.body.position.x), float(p.body.position.y)],
                "vel": [float(p.body.velocity.x), float(p.body.velocity.y)],
            })

    # --- 锁外：轻量积分 ---
    G0 = float(G)

    # 保护一下输入，避免有人 seconds=1e9 来挖矿
    seconds = max(1.0, min(seconds, 600.0))
    dt = max(1.0/240.0, min(dt, 1.0/5.0))   # 过小过大都不划算
    steps = int(max(1, seconds / dt))

    # 可选：对点数做硬上限，避免前端/网络爆炸
    max_pts = 2000
    est_pts = steps // stride + 1
    if est_pts > max_pts:
        stride = max(1, int(math.ceil(steps / (max_pts - 1))))

    def planet_planet_acc(i):
        """返回第 i 颗行星受到其他行星引力产生的加速度 (ax, ay)"""
        xi, yi = planets[i]["pos"]
        mi = planets[i]["mass"]
        ax = ay = 0.0
        for j in range(len(planets)):
            if j == i:
                continue
            xj, yj = planets[j]["pos"]
            mj = planets[j]["mass"]
            dx = xj - xi
            dy = yj - yi
            r2 = dx*dx + dy*dy
            if r2 < 1e-12:
                continue
            r = math.sqrt(r2)
            inv_r3 = 1.0 / (r2 * r)
            a = G0 * mj * inv_r3
            ax += dx * a
            ay += dy * a
        return ax, ay

    def ship_acc_from_planets(x, y):
        """core 在 (x,y) 处受到行星引力的加速度 (ax, ay)，用 eff_r=max(dist,R) 做软化"""
        ax = ay = 0.0
        for pl in planets:
            px, py = pl["pos"]
            dx = px - x
            dy = py - y
            r2 = dx*dx + dy*dy
            if r2 < 1e-12:
                continue
            r = math.sqrt(r2)
            eff_r = max(r, pl["radius"])
            eff_r2 = eff_r * eff_r
            eff_r3 = eff_r2 * eff_r
            a = G0 * pl["mass"] / eff_r3
            ax += dx * a
            ay += dy * a
        return ax, ay

    # 状态
    sx, sy = core_pos
    svx, svy = core_vel

    points = []
    for step in range(steps + 1):
        if step % stride == 0:
            points.append({"x": float(sx), "y": float(sy)})

        # 1) 行星自己动（如果你不想要月球跑，就把这段注释掉）
        # 半隐式：v += a*dt; x += v*dt
        # 行星数量通常很少，这点开销几乎等于 0
        p_acc = [planet_planet_acc(i) for i in range(len(planets))]
        for i, (ax, ay) in enumerate(p_acc):
            planets[i]["vel"][0] += ax * dt
            planets[i]["vel"][1] += ay * dt
        for i in range(len(planets)):
            planets[i]["pos"][0] += planets[i]["vel"][0] * dt
            planets[i]["pos"][1] += planets[i]["vel"][1] * dt

        # 2) 飞船 core 在行星引力下动
        ax, ay = ship_acc_from_planets(sx, sy)
        svx += ax * dt
        svy += ay * dt
        sx += svx * dt
        sy += svy * dt

    return jsonify({
        "ok": True,
        "ship_id": int(req_ship_id),
        "dt": dt,
        "seconds": seconds,
        "stride": stride,
        "mode": mode,
        "points": points,
    })

# ------------------------------
# main
# ------------------------------
if __name__ == "__main__":
    init_world()
    t = threading.Thread(target=Physics_loop, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=5050, debug=False, use_reloader=False)
