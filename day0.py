import marimo

__generated_with = "0.18.3"
app = marimo.App(width="medium")


@app.cell
def _():
    from sandbox import DockerSandbox, Environment
    return DockerSandbox, Environment


@app.cell
def _():
    import os
    return


@app.cell
def _(Environment):
    async def run():
        """Use the custom sandbox."""
        async with Environment() as env:
            sandbox = env.create_sandbox(
                type="docker",
                command="sleep",
                args=["infinity"],
                container_image="python:3.11-slim",
            )

            async with sandbox:
                result = await sandbox.exec(["echo", "Hello"])
                print(f"Result: {result.stdout} | Return code: {result.returncode}")
    return


@app.cell
def _():
    # await run()
    return


@app.cell
def _(Environment):
    environ = Environment()

    sandbox = environ.create_sandbox(
        type="docker",
        command="sleep",
        args=["infinity"],
        container_image="python:3.11-slim",
    )

    # await sandbox.exec(['echo', 'hello'])
    return (sandbox,)


@app.cell
async def _(sandbox):
    await sandbox.start(command="sleep", args=["infinity"], container_image='python:3.11-slim')
    return


@app.cell
async def _(sandbox):
    await sandbox.exec(['echo', 'hello'])
    return


@app.cell
async def _(sandbox):
    await sandbox.write_file(
        filepath="./z.md",
        contents="# HELLO\nThis is an example zubin file\n".encode('utf-8')
    )
    return


@app.cell
async def _(sandbox):
    await sandbox.read_file('./z.md')
    return


@app.cell
async def _(DockerSandbox):
    dockersand = DockerSandbox(container_image="python:3.11-slim", command="sleep")
    await dockersand.start(command="sleep", args=["infinity"], container_image="python:3.11-slim")
    return (dockersand,)


@app.cell
async def _(sandbox):
    await sandbox.stop()
    return


@app.cell
async def _(dockersand):
    await dockersand.exec(command=['touch','zz.md'])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
