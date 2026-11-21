"""One-shot diagnostic: extract the <script> body from the served /webcam
HTML and sanity-check it for common failure modes."""
import re

html = open('_webcam_served.html', encoding='utf-8').read()
s = html.find('<script>') + len('<script>')
e = html.find('</script>', s)
js = html[s:e]
print(f'JS len: {len(js)}')
open('_dodo.js', 'w', encoding='utf-8').write(js)

print('open braces:', js.count('{'), 'close braces:', js.count('}'))
print('open parens:', js.count('('), 'close parens:', js.count(')'))
print('open brackets:', js.count('['), 'close brackets:', js.count(']'))
print('template backticks:', js.count('`'))

# Count suspicious escape sequences that could break JS parse
for label, pat in [('double-backslash', r'\\\\'), ('\\n escape', r'\\n'),
                   ('\\t escape', r'\\t'), ('\\r escape', r'\\r')]:
    c = len(re.findall(pat, js))
    print(f'{label:<20} count={c}')

# Dump the first 20 lines and last 20 lines for eyeball
lines = js.splitlines()
print('\n-- FIRST 10 LINES --')
for i, ln in enumerate(lines[:10], 1):
    print(f'{i:3}: {ln[:160]}')
print('\n-- LAST 10 LINES --')
for i, ln in enumerate(lines[-10:], len(lines) - 9):
    print(f'{i:3}: {ln[:160]}')

# Hunt for any line that contains a stray backslash outside a string or regex
# Rough scan: print lines that contain `\\\\` (served as `\\` in JS)
print('\n-- LINES WITH DOUBLE-BACKSLASHES (potential problem) --')
for i, ln in enumerate(lines, 1):
    if '\\\\' in ln:
        print(f'{i:3}: {ln[:200]}')
