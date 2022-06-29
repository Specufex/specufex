# noxfile.py
import nox

locations = "specufex", "tests", "noxfile.py"
nox.options.sessions = "lint", "tests"


@nox.session(python=["3.10", "3.9", "3.8"], venv_backend="mamba")
def tests(session):
    session.run("mamba", "install", "-y", "--file", "requirements.txt")
    session.run("pip", "install", "-r", "requirements-dev.txt")
    session.install("-e", ".", "--no-deps")
    session.run("pytest", "--cov")


@nox.session(python=["3.10", "3.9", "3.8"], venv_backend="mamba")
def lint(session):
    args = session.posargs or locations
    session.install(
        "flake8",
        "flake8-black",
        "flake8-isort",
        "flake8-pyprojecttoml",
    )
    session.run("flake8", *args)


@nox.session(python=["3.10", "3.9", "3.8"], venv_backend="mamba")
def black(session):
    args = session.posargs or locations
    session.install("black")
    session.run("black", *args)
