import re

s = open("main.md").read()

s = re.sub(r"<script[\s\S]*?</script>", r'', s)
s = re.sub(r"<style>[\s\S]*?</style>", r'', s)

for i in range(10):
    s = re.sub(r"<details[\s\S]*?>([\s\S]*?)<summary>([\s\S]*?)</summary>([\s\S]*?)</details>", r'<spoiler summary="\g<2>"> \g<1> \g<3> </spoiler>', s)

s = re.sub(r"<[/]{0,1}blockquote>", r'', s)
s = re.sub(r'src="([\s\S]*?)"', r'src="https://raw.githubusercontent.com/piskareviv/blog_aux_ntt_1/f24106fac047304d192ccbc1b89b6b324f307770/\g<1>"', s)
s = re.sub(r"\[<span[\s\S]*?</span>\]\(https://codeforces.com/profile/([\s\S]*?)\)", r'[user:\g<1>]', s)


s = re.sub(r"with avx2\n", r"with avx2\n\n" + "> Text is also available on [github pages](https://piskareviv.github.io/blog_aux_ntt_1/)\n", s)


open("main.cf", 'w').write(s)
