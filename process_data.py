import os
import RNA
import glob

def process_file(in_filepath, out_filepath):
    with open(in_filepath, 'r') as f:
        lines = f.readlines()
    
    out_lines = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        
        # header
        if line.startswith(">"):
            header = line
            
            i += 1
            if i >= len(lines):
                break
            seq = lines[i].strip()
            
            # calculate secondary structure and mfe
            struct, mfe = RNA.fold(seq.upper())
            
            out_lines.append(header + "\n")
            out_lines.append(seq + "\n")
            out_lines.append(f"{struct} (MFE: {mfe:.2f} kcal/mol)\n")
        i += 1

    with open(out_filepath, 'w') as f:
        f.writelines(out_lines)
    print(f"Processed: {in_filepath} -> {out_filepath}")

def main():
    data_dir = "data"
    out_dir = "data-processed"
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    # iter files
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".fa"):
                in_filepath = os.path.join(root, file)
                
                # compute output path
                rel_path = os.path.relpath(in_filepath, data_dir)
                out_filepath = os.path.join(out_dir, rel_path)
                
                # ensure outdir exists
                os.makedirs(os.path.dirname(out_filepath), exist_ok=True)
                
                process_file(in_filepath, out_filepath)
                
if __name__ == "__main__":
    main()
