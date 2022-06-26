# noxfile.py
import nox
@nox.session(python=["3.10", "3.9", "3.8"], venv_backend="mamba")
def tests(session):
    session.run("mamba", "install", "-y", "--file", "requirements.txt")
    session.install(".", "--no-deps")
    session.run("pytest")