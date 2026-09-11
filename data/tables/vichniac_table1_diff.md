# Vichniac (1990), Table 1: entries that differ from the computed gradient

Computed from the definition of the Boolean derivative, eq. (5) of the
paper, exhaustively over the eight neighbourhoods; compared by truth
table. Patterns are over the two remaining variables in the order
(1,1), (1,0), (0,1), (0,0). Notation: L = x[i-1], C = x[i], R = x[i+1];
lower case = complemented literal; juxtaposition = AND; + = OR.

## Rule 62 (00111110), d/dx[i-1]

- printed expression: `c+R`
- printed pattern:    `1011`
- correct pattern:    `1101`
- correct expression: `C+r`

## Rule 110 (01101110), d/dx[i-1]

- printed expression: `cR`
- printed pattern:    `0010`
- correct pattern:    `1000`
- correct expression: `CR`

## Rule 110 (01101110), d/dx[i+1]

- printed expression: `L+C`
- printed pattern:    `1110`
- correct pattern:    `1101`
- correct expression: `L+c`

## Rule 130 (10000010), d/dx[i+1]

- printed expression: `RC+rc`
- printed pattern:    `0101`  (illegal: contains the variable it is differentiated with respect to; shown with that variable set to 0)
- correct pattern:    `1001`
- correct expression: `LC+lc`

## Rule 146 (10010010), d/dx[i-1]

- printed expression: `C+R`
- printed pattern:    `1110`
- correct pattern:    `1011`
- correct expression: `R+c`

## Rule 146 (10010010), d/dx[i+1]

- printed expression: `L+C`
- printed pattern:    `1110`
- correct pattern:    `1101`
- correct expression: `L+c`

## Rule 172 (10101100), d/dx[i-1]

- printed expression: `CR+cr`
- printed pattern:    `1001`
- correct pattern:    `0110`
- correct expression: `Cr+Rc`

7 mismatching entries in 5 rules: 62, 110, 130, 146, 172
