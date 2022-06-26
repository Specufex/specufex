# noxfile.py
import nox

locations = "specufex", "tests", "noxfile.py"


@nox.session(python=["3.10", "3.9", "3.8"], venv_backend="mamba")
def tests(session):
    session.run("mamba", "install", "-y", "--file", "requirements.txt")
    session.install(".", "--no-deps")
    session.run("pytest")


@nox.session(python=["3.10", "3.9", "3.8"], venv_backend="mamba")
def lint(session):
    args = session.posargs or locations
    session.install("flake8")
    session.run("flake8", *args)


@nox.session(python=["3.10", "3.9", "3.8"], venv_backend="mamba")
def black(session):
    args = session.posargs or locations
    session.install("black")
    session.run("black", *args)
