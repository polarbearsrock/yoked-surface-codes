import stim

from gen._util import estimate_qubit_count_during_postselection, write_file


def test_write_file_bytes_round_trip(tmp_path):
    path = tmp_path / 'data.bin'
    content = b'\x00\x01binary \xff bytes\n'
    write_file(path, content)
    assert path.read_bytes() == content


def test_write_file_str(tmp_path):
    path = tmp_path / 'data.txt'
    write_file(path, 'text content')
    # The str branch prints into the file, which appends a newline.
    assert path.read_text() == 'text content\n'


def test_estimate_qubit_count_during_postselection():
    assert estimate_qubit_count_during_postselection(stim.Circuit("""
        QUBIT_COORDS(0, 0) 100
        H 55
        M 55
    """)) == 0

    assert estimate_qubit_count_during_postselection(stim.Circuit("""
        QUBIT_COORDS(0, 0) 100
        H 55
        M 55
        DETECTOR(0, 0, 0, 999) rec[-1]
    """)) == 1

    assert estimate_qubit_count_during_postselection(stim.Circuit("""
        QUBIT_COORDS(0, 0) 100
        H 55 56
        M 55
        DETECTOR(0, 0, 0, 999) rec[-1]
    """)) == 2

    assert estimate_qubit_count_during_postselection(stim.Circuit("""
        QUBIT_COORDS(0, 0) 100
        H 55 56
        M 55
        DETECTOR(0, 0, 0, 999) rec[-1]
        H 57
    """)) == 2

    assert estimate_qubit_count_during_postselection(stim.Circuit("""
        QUBIT_COORDS(0, 0) 100
        H 55 56
        M 55
        REPEAT 10 {
            H 58
        }
        DETECTOR(0, 0, 0, 999) rec[-1]
        H 57
    """)) == 3
