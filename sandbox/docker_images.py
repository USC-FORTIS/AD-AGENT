import os
import subprocess

# Maps package_name -> (image_tag, dockerfile_path relative to project root)
DOCKER_IMAGE_MAP = {
    "pyod":  ("openad-pyod:latest",  "sandbox/dockerfiles/Dockerfile.pyod"),
    "pygod": ("openad-pygod:latest", "sandbox/dockerfiles/Dockerfile.pygod"),
    "darts": ("openad-darts:latest", "sandbox/dockerfiles/Dockerfile.darts"),
    "tslib": ("openad-tslib:latest", "sandbox/dockerfiles/Dockerfile.tslib"),
    "tsb_ad": ("openad-tsb-ad:latest", "sandbox/dockerfiles/Dockerfile.tsb_ad"),
}


def ensure_image(package_name: str) -> str:
    """
    Return the Docker image tag for *package_name*, building it first if the
    image does not already exist locally.
    """
    entry = DOCKER_IMAGE_MAP.get(package_name)
    if entry is None:
        raise ValueError(f"Unknown package for Docker: {package_name}")

    image_tag, dockerfile_path = entry

    # Check if the image already exists
    result = subprocess.run(
        ["docker", "image", "inspect", image_tag],
        capture_output=True,
    )
    if result.returncode == 0:
        return image_tag

    # Resolve dockerfile path relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    abs_dockerfile = os.path.join(project_root, dockerfile_path)

    print(f"[Docker] Building image {image_tag} from {dockerfile_path} ...")
    build = subprocess.run(
        ["docker", "build", "-t", image_tag, "-f", abs_dockerfile, project_root],
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        raise RuntimeError(
            f"Docker build failed for {image_tag}:\n{build.stderr}"
        )
    print(f"[Docker] Image {image_tag} built successfully.")
    return image_tag
