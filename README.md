# surforama
a napari-based tool for using surfaces to explore volumetric data in napari

inspired by [membranorama](https://github.com/dtegunov/membranorama)

![Screenshot of surforama showing a surface in the slice of a tomogram](surforama_screenshot.png)

## installation
First, install napari. Then install surforama in the same environment.

```python
pip install surforama
```

## usage
### launch without data
You can launch surforama using the command line interface. After you have installed surforama, you can launch it with the following command in your terminal:

```bash
surforama
```
After surforama launches, you can load your image and mesh into napari and get surfing!

### launch with data
If you have an MRC-formatted tomogram and an obj-formatted mesh, you can launch using the following command:

```bash
surforama --image-path /path/to/image.mrc --mesh-path /path/to/mesh.obj
```

## developer installation

If you would like to make changes to the surforama source code, you can install surformama with the developer tools as follows:

```bash
cd /path/to/your/surforama/source/code/folder
pip install -e ".[dev]"
```
We use pre-commit to keep the code tidy. Install the pre-commit hooks to activate the checks:

```bash
pre-commit install
```
