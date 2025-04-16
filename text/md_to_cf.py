import re

s = open("main.md").read()

s = re.sub(r"<details>([\s\S]*?)<summary>([\s\S]*?)</summary>([\s\S]*?)</details>", r'<spoiler summary="\g<2>"> \g<1> \g<3> </spoiler>', s)
s = re.sub(r"<[/]{0,1}blockquote>", r'', s)

open("main.cf", 'w').write(s)
