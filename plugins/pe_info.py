"""
pe_info.py

A simple plugin to extract and display key information from a Windows PE file.
Depends on: pefile (pip install pefile)
"""

import pefile
import os

def analyze_pe(path):
    """
    Load a PE file and print out its COFF header, optional header, and sections.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such file: {path}")

    pe = pefile.PE(path, fast_load=True)
    pe.parse_data_directories(directories=[
        pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IMPORT'],
        pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_EXPORT'],
        pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_RESOURCE']
    ])

    print(f"== PE FILE: {os.path.basename(path)} ==")
    print_header(pe)
    print_sections(pe)
    print_imports(pe)
    print_exports(pe)

def print_header(pe):
    """
    Print COFF and Optional headers.
    """
    coff = pe.FILE_HEADER
    opt = pe.OPTIONAL_HEADER

    print("\n-- COFF FILE HEADER --")
    print(f"Machine: 0x{coff.Machine:04X}")
    print(f"NumberOfSections: {coff.NumberOfSections}")
    print(f"TimeDateStamp: {coff.TimeDateStamp}")

    print("\n-- OPTIONAL HEADER --")
    print(f"EntryPoint: 0x{opt.AddressOfEntryPoint:08X}")
    print(f"ImageBase: 0x{opt.ImageBase:016X}")
    print(f"Subsystem: {opt.Subsystem}")
    print(f"SizeOfImage: {opt.SizeOfImage}")

def print_sections(pe):
    """
    List all sections with their virtual addresses and sizes.
    """
    print("\n-- SECTIONS --")
    for sec in pe.sections:
        name = sec.Name.rstrip(b'\x00').decode('utf-8', errors='ignore')
        print(f"{name:<8} VA:0x{sec.VirtualAddress:08X} VS:0x{sec.Misc_VirtualSize:08X}")

def print_imports(pe):
    """
    List DLLs and the imported symbols.
    """
    print("\n-- IMPORTS --")
    if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            dll = entry.dll.decode()
            print(f"{dll}:")
            for imp in entry.imports:
                name = imp.name.decode() if imp.name else f"Ordinal_{imp.ordinal}"
                print(f"  - {name}")
    else:
        print("  <none>")

def print_exports(pe):
    """
    List the exported functions, if any.
    """
    print("\n-- EXPORTS --")
    if hasattr(pe, 'DIRECTORY_ENTRY_EXPORT'):
        for exp in pe.DIRECTORY_ENTRY_EXPORT.symbols:
            name = exp.name.decode() if exp.name else f"Ordinal_{exp.ordinal}"
            print(f"  - {name} (RVA: 0x{exp.address:08X}, Ordinal: {exp.ordinal})")
    else:
        print("  <none>")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <pe_file_path>")
        sys.exit(1)

    analyze_pe(sys.argv[1])
