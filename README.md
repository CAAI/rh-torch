# rh-torch
Shared scripts for training deep learning networks in torch 

## HOW TO INSTALL

Install only to your user. Go to your virtual environment. Run:
```
git clone https://github.com/CAAI/rh-torch.git && cd rh-torch
pip install .
```

## HOW TO CONTRIBUTE

Create your edits in a different branch, decicated to a few specific things. We prefer many minor edits over one huge. Once everything is well-documented and tested, perform a pull request for your edits to be made available in the main branch. See steps here:
```
git branch awesome-addition
git checkout awesome-addition
# do your changes
git commit -a -m 'your changes'
git push
gh pr create --title "your title" --body "longer description of what you did"
```
If you wish, you can add ```--assignee <github_name>``` to ping specific persons for looking at your pull request.

One someone accepted your pull request (after reviewing the changes), it will be part of the main branch.

