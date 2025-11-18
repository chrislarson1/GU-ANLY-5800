# Cloud Compute Access for Final Projects

Feel free to use any cloud resources at your disposal for the final project. One option is to use free resources provided to us by Jetstream 2. To get setup on JS2, please schedule a 30 minute meeting with me to setup these resources.

## Jetstream 2 Access Instructions

Please read through the following instructions carefully before accessing the Jupyter Notebook environment. If you are uncertain about any of the instructions, please reach out to the instructor via Discord or email.

### Instance Specs

Each group will has a dedicated Jupyter Notebook container with the following specifications:

| Resource | Specification |
|----------|---------------|
| CPU | 18 cores |
| Memory | 64 GB |
| Storage | 250 GB |
| GPU | 1 x 40 GB VRAM (A100) |

### Access Instructions

To access our hosted jupyter environment, you'll need to install [kubectl](https://kubernetes.io/docs/tasks/tools/) on your local machine. For Windows users, you'll also need to install [docker](https://docs.docker.com/desktop/install/windows-install/).

###### Unix based operating systems
To connect to your Jupyter instance, run the following command, replacing the arguments with your group name and port number (optional, defaults to 5800).

```bash
bash js2.sh <group-name> <port>
```

As an example, if you are in group 3, you will replace `<group-name>` with `group-3`, and then you can access your jupyter environment [here](http://localhost:5800). When you are finished with your session, make sure to `ctrl-c` to close the connection.

###### Windows operating systems
First build the docker image:

```bash
bash js2-docker-build.sh
```

You can now connect to your cluster with this command:

```bash
bash js2-docker.sh <group-name> <port>
```

and make sure to replace `<group-name` with the name of your group (e.g., `group-2`) and `<port>` with a specific port number if the default port of 5800 is already in use.

### Storage

Once you're logged in, the root directory will be `/workspace`. Your persistent volume is mounted to `/workspace/anly5800`. **Make sure to store any data, models, and artifacts that you want saved under that directory!** Importantly, when downloading datasets and models using third part tools (e.g., huggingface), you'll need to make sure to pass in the appropriate flags so that that those artifacts are saved to that directory, and not under the default path, which may not be inside of your attached volume and thus will have limited disk space.

### Python Environment

The jupyter servers come preconfigured with Torch. I highly recommend that you not configure the base python environment directly, but rather always use a sandbox environment (or multiple of them if needed).

To create individual virtual environments using virtual env first do:

```bash
mkdir /workspace/anly5800/sandbox/
```

and then within that folder create a virtual environment

```bash
python -m venv .venv && source .venv/bin/activate
```

Now you can install sandbox-specific packages.
