#!/bin/bash
# Monitor R2 job progress
echo "=== R2 Job Status: $(date) ==="
echo ""

for f in /home/woodbuse/codebase_projects/quantum_cowboy_biochemistry/logs/R2_*.stderr; do
    name=$(basename "$f" .stderr | sed 's/R2_//;s/_1220[0-9]*//' | cut -c1-50)
    last=$(tail -1 "$f" 2>/dev/null)
    
    # Extract key progress info
    if echo "$last" | grep -q "Barrier"; then
        status="DONE"
    elif echo "$last" | grep -q "NEB"; then
        status="NEB "
    elif echo "$last" | grep -q "MD"; then
        status="MD  "
    elif echo "$last" | grep -q "relax"; then
        status="RELX"
    else
        status="????"
    fi
    
    echo "[$status] $name"
    echo "        $last"
    echo ""
done

# Check if any jobs completed
n_done=$(squeue -u woodbuse -h 2>/dev/null | wc -l)
echo "Active jobs: $n_done / 8"
