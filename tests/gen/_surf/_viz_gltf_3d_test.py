import base64
import json

import numpy as np
import pygltflib

from gen._surf._viz_gltf_3d import ColoredLineData, ColoredTriangleData, gltf_model_from_colored_triangle_data, viz_3d_gltf_model_html


def test_gltf_model_from_colored_triangle_data():
    model = gltf_model_from_colored_triangle_data([
        ColoredTriangleData(
            rgba=(1, 0, 0, 1),
            triangle_list=np.array(
                [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            ),
        ),
        ColoredTriangleData(
            rgba=(1, 0, 1, 1),
            triangle_list=np.array(
                [
                    [1, 1, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            ),
        ),
    ])
    assert json.loads(model.to_json()) == {
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126,
                "normalized": False,
                "count": 3,
                "type": "VEC3",
                "max": [
                    1.0,
                    1.0,
                    1.0
                ],
                "min": [
                    0.0,
                    0.0,
                    0.0
                ]
            },
            {
                "bufferView": 1,
                "byteOffset": 0,
                "componentType": 5126,
                "normalized": False,
                "count": 3,
                "type": "VEC3",
                "max": [
                    1.0,
                    1.0,
                    1.0
                ],
                "min": [
                    0.0,
                    0.0,
                    0.0
                ]
            }
        ],
        "asset": {
            "generator": f"pygltflib@v{pygltflib.__version__}",
            "version": "2.0"
        },
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": 36,
                "target": 34962
            },
            {
                "buffer": 0,
                "byteOffset": 36,
                "byteLength": 36,
                "target": 34962
            }
        ],
        "buffers": [
            {
                "uri": "data:application/octet-stream;base64,AACAPwAAAAAAAAAAAAAAAAAAgD8AAAAAAAAAAAAAAAAAAIA/AACAPwAAgD8AAAAAAAAAAAAAgD8AAAAAAAAAAAAAAAAAAIA/",
                "byteLength": 72
            }
        ],
        "materials": [
            {
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1, 0, 0, 1],
                    "metallicFactor": 0.3,
                    "roughnessFactor": 0.8
                },
                "doubleSided": True
            },
            {
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1, 0, 1, 1],
                    "metallicFactor": 0.3,
                    "roughnessFactor": 0.8
                },
                "doubleSided": True
            }
        ],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": 0
                        },
                        "mode": 4,
                        "material": 0
                    },
                    {
                        "attributes": {
                            "POSITION": 1
                        },
                        "mode": 4,
                        "material": 1
                    }
                ]
            }
        ],
        "nodes": [
            {
                "mesh": 0
            }
        ],
        "scenes": [
            {
                "nodes": [
                    0
                ]
            }
        ]
    }


def test_viz_3d_gltf_model_html():
    model = gltf_model_from_colored_triangle_data([
        ColoredTriangleData(
            rgba=(1, 0, 0, 1),
            triangle_list=np.array(
                [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            ),
        ),
        ColoredTriangleData(
            rgba=(1, 0, 1, 1),
            triangle_list=np.array(
                [
                    [1, 1, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            ),
        ),
    ])

    html = viz_3d_gltf_model_html(model)
    assert '<html>' in html


def test_gltf_model_with_same_rgba_triangle_and_line_groups():
    rgba = (1, 0, 0, 1)
    model = gltf_model_from_colored_triangle_data(
        [
            ColoredTriangleData(
                rgba=rgba,
                triangle_list=np.array(
                    [
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                    ],
                    dtype=np.float32,
                ),
            ),
        ],
        colored_line_data=[
            ColoredLineData(
                rgba=rgba,
                edge_list=np.array(
                    [
                        [0, 0, 0],
                        [1, 1, 1],
                    ],
                    dtype=np.float32,
                ),
            ),
        ],
    )
    d = json.loads(model.to_json())
    tri_prim, line_prim = d["meshes"][0]["primitives"]
    assert tri_prim["mode"] == pygltflib.TRIANGLES
    assert line_prim["mode"] == pygltflib.LINES

    # Sharing an rgba between a triangle group and a line group must not
    # collide: each group keeps its own material, buffer view, and accessor.
    assert tri_prim["material"] != line_prim["material"]
    tri_acc = d["accessors"][tri_prim["attributes"]["POSITION"]]
    line_acc = d["accessors"][line_prim["attributes"]["POSITION"]]
    assert tri_acc["count"] == 3
    assert line_acc["count"] == 2
    assert tri_acc["bufferView"] != line_acc["bufferView"]
    assert d["bufferViews"][tri_acc["bufferView"]] == {
        "buffer": 0,
        "byteOffset": 0,
        "byteLength": 36,
        "target": 34962,
    }
    assert d["bufferViews"][line_acc["bufferView"]] == {
        "buffer": 0,
        "byteOffset": 36,
        "byteLength": 24,
        "target": 34962,
    }
    buffer = d["buffers"][0]
    encoded_data = buffer["uri"].split(",", maxsplit=1)[1]
    assert buffer["byteLength"] == len(base64.b64decode(encoded_data)) == 60
