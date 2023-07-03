## ODT case input files

These folders provide examples of ODT cases. The key input file for a given case is `input.yaml`. The `gas_mechanisms` case is not an input case, but consists of Cantera xml mechanism files that correspond to the `chemMechFile` field of the input file.

### DumpTimes generation for input.yaml file (Vim Tips): 
The dumpTimes variable of an input.yaml is a list of discrete times, that for the yaml file each time ocupies a new line, and is preceded by "   - " as:
```
   - 50.0
   - 50.2
   - 50.4
   etc.
```
To generate a sequence of dumpTimes from 50.0 to 500.0 with steps of 0.2 in an input.yaml file using `vim`, locate the cursor after `dump Times:` line an run the commands:
```bash
# -> deletes everything from cursor line
Esc + dG 
# -> write times secuence, single time for each line
:put=map(range(500,5001,2), 'v:val/10.0')
# -> go to file begining
gg
# -> move cursor until first time line
down-arrow until first dumpTimes line
# -> Select all dumpTimes lines
Ctrl v + G + I +    - + Esc
# where Ctrl+v activates 'VISUAL BLOCK' mode, "G" or "shift g" goes to last line of the file, "I" or "shift i" activates visual block Insert mode, "   - " sequence of characters to insert (you see it as you edit only first selected line), and "Esc" copies the inserted characters sequence to all selected lines.
# -> save + exit 
:wq
```  