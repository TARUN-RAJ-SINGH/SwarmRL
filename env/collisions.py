"""
SwarmRL — Collision Detection System

Handles:
  • Drone ↔ Drone collisions (sphere–sphere)
  • Drone ↔ Obstacle collisions (sphere–AABB)

Returns collision events that the environment uses for penalties and physics.
"""

import numpy as np
from env.world_config import WORLD_CONFIG


def check_drone_drone_collisions(drones: list) -> list[tuple[int, int]]:
    """
    Check all pairs of alive drones for sphere–sphere overlap.

    Returns:
        List of (drone_i.id, drone_j.id) tuples that are colliding.
    """
    collisions = []
    alive = [d for d in drones if d.alive]

    for i in range(len(alive)):
        for j in range(i + 1, len(alive)):
            dist = np.linalg.norm(alive[i].position - alive[j].position)
            min_dist = alive[i].radius + alive[j].radius
            if dist < min_dist:
                collisions.append((alive[i].id, alive[j].id))
    return collisions


def check_drone_obstacle_collisions(drones: list, obstacles: list = None) -> list[tuple[int, str]]:
    """
    Check each alive drone against every obstacle (sphere vs AABB).

    Returns:
        List of (drone_id, obstacle_id) tuples that are colliding.
    """
    if obstacles is None:
        obstacles = WORLD_CONFIG["obstacles"]

    collisions = []
    for drone in drones:
        if not drone.alive:
            continue
        for obs in obstacles:
            obs_pos = np.array(obs["position"])
            obs_size = np.array(obs["size"])

            # Nearest point on AABB to drone center
            closest = np.clip(drone.position, obs_pos, obs_pos + obs_size)
            dist = np.linalg.norm(drone.position - closest)

            if dist < drone.radius:
                collisions.append((drone.id, obs["id"]))

    return collisions


def resolve_drone_drone_collision(drone_a, drone_b):
    """
    Simple elastic-style resolution: push drones apart along
    the collision axis and reflect velocities.
    """
    diff = drone_a.position - drone_b.position
    dist = np.linalg.norm(diff)
    min_dist = drone_a.radius + drone_b.radius

    if dist < 1e-6:
        # Drones are exactly on top of each other — push apart randomly
        diff = np.random.randn(3)
        dist = np.linalg.norm(diff)

    normal = diff / dist
    overlap = min_dist - dist

    # Separate equally
    drone_a.position += normal * (overlap / 2 + 0.01)
    drone_b.position -= normal * (overlap / 2 + 0.01)

    # Reflect velocities along the collision normal
    drone_a.velocity -= normal * np.dot(drone_a.velocity, normal)
    drone_b.velocity -= normal * np.dot(drone_b.velocity, normal)

    drone_a.collisions += 1
    drone_b.collisions += 1


def resolve_drone_obstacle_collision(drone, obstacle: dict):
    """
    Push the drone out of the obstacle AABB and zero velocity
    along the penetration axis.
    """
    obs_pos = np.array(obstacle["position"])
    obs_size = np.array(obstacle["size"])

    closest = np.clip(drone.position, obs_pos, obs_pos + obs_size)
    diff = drone.position - closest
    dist = np.linalg.norm(diff)

    if dist < 1e-6:
        # Drone center is inside the AABB — push to nearest face
        half = obs_size / 2.0
        center = obs_pos + half
        to_center = drone.position - center
        # Find the axis with smallest penetration depth
        penetration = half - np.abs(to_center)
        axis = int(np.argmin(penetration))
        direction = np.sign(to_center[axis]) if to_center[axis] != 0 else 1.0
        drone.position[axis] = (
            center[axis] + direction * (half[axis] + drone.radius + 0.01)
        )
        drone.velocity[axis] = 0.0
    else:
        normal = diff / dist
        push = drone.radius - dist + 0.01
        drone.position += normal * push
        drone.velocity -= normal * np.dot(drone.velocity, normal)

    drone.collisions += 1
