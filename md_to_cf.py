import re

s = open("main.md").read()

s = re.sub(r"<details>([\s\S]*?)<summary>([\s\S]*?)</summary>([\s\S]*?)</details>", r'<spoiler summary="\g<2>"> \g<1> \g<3> </spoiler>', s)
s = re.sub(r"<[/]{0,1}blockquote>", r'', s)
s = re.sub(r'src="([\s\S]*?)"', r'src="https://raw.githubusercontent.com/piskareviv/blog_aux_ntt_1/refs/heads/master/\g<1>"', s)

open("main.cf", 'w').write(s)
