
import pytest
import stim

import gen


def test_chunk_normalizes_flows_and_discards_to_tuples():
    chunk = gen.Chunk(
        circuit=stim.Circuit("""
            R 0
        """),
        q2i={0: 0},
        flows=[],
        discarded_inputs=[gen.PauliString({0: 'Z'})],
        discarded_outputs=[gen.PauliString({0: 'X'})],
    )
    assert isinstance(chunk.flows, tuple)
    assert isinstance(chunk.discarded_inputs, tuple)
    assert isinstance(chunk.discarded_outputs, tuple)


def test_verify_rejects_duplicate_flow_starts():
    chunk = gen.Chunk(
        circuit=stim.Circuit("""
            M 0
        """),
        q2i={0: 0},
        flows=[
            gen.Flow(center=0, start=gen.PauliString({0: 'Z'}), measurement_indices=[0]),
            gen.Flow(center=0, start=gen.PauliString({0: 'Z'}), measurement_indices=[0]),
        ],
    )
    with pytest.raises(ValueError, match="same non-empty start"):
        chunk.verify()


def test_inverse_flows():
    chunk = gen.Chunk(
        circuit=stim.Circuit("""
            R 0 1 2 3 4
            CX 2 0
            M 0
        """),
        q2i={0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
        flows=[
            gen.Flow(
                center=0,
                start=gen.PauliString({}),
                measurement_indices=[0],
                end=gen.PauliString({1: 'Z'}),
            ),
        ],
    )

    inverted = chunk.inverted()
    inverted.verify()
    assert len(inverted.flows) == len(chunk.flows)
    assert inverted.circuit == stim.Circuit("""
        R 0
        CX 2 0
        M 4 3 2 1 0
    """)


def test_inverse_circuit():
    chunk = gen.Chunk(
        circuit=stim.Circuit("""
            R 0 1 2 3 4
            CX 2 0 3 4
            X 1
            M 0
        """),
        q2i={0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
        flows=[],
    )

    inverted = chunk.inverted()
    inverted.verify()
    assert len(inverted.flows) == len(chunk.flows)
    assert inverted.circuit == stim.Circuit("""
        R 0
        X 1
        CX 3 4 2 0
        M 4 3 2 1 0
    """)


def test_with_flows_postselected():
    chunk = gen.Chunk(
        circuit=stim.Circuit("""
            R 0
        """),
        q2i={0: 0},
        flows=[gen.Flow(
            center=0,
            end=gen.PauliString({0: 'Z'}),
        )],
    )
    assert chunk.with_flows_postselected(lambda f: False) == chunk
    assert chunk.with_flows_postselected(lambda f: True) != chunk
    assert chunk.with_flows_postselected(lambda f: f.center == 1) == chunk
    assert chunk.with_flows_postselected(lambda f: f.center == 0) == gen.Chunk(
        circuit=stim.Circuit("""
            R 0
        """),
        q2i={0: 0},
        flows=[gen.Flow(
            center=0,
            end=gen.PauliString({0: 'Z'}),
        ).postselected()],
    )
