# noxfile.py
import nox
@nox.session(python=["3.10", "3.9", "3.8"])
def tests(session):
    session.run("pytest", "--cov")