#!/bin/bash
# Create a github_repo from the command line...so cool (more here: https://developer.github.com/v3/repos/#create)
# I should probably implement some check if the creation was successful
curl -u 'AleksiNummelin' https://api.github.com/user/repos -d '{"name":"scale_dependent_diffusivity", "private":false}'

# Link local repository to git
git init
git add *
git add .gitignore .stickler.yml .travis.yml
git commit -m 'first commit'
git remote add origin git@github.com:AleksiNummelin/scale_dependent_diffusivity.git
git push -u origin master
