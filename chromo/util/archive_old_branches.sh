# This script was found online: https://stackoverflow.com/questions/1307114/how-can-i-archive-git-branches

git checkout -b BRANCH_NAME origin/BRANCH_NAME
git tag archive/BRANCH_NAME BRANCH_NAME
git checkout master
git branch -D BRANCH_NAME
git branch -d -r origin/BRANCH_NAME
git push --tags
git push origin :BRANCH_NAME