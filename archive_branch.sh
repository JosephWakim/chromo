# Archive old branches on a GitHub repository (for housekeeping purposes)

# This script was derived from the following StackOverflow post:
# https://stackoverflow.com/questions/1307114/how-can-i-archive-git-branches

# Usage:
# $ bash archive_branch.sh BRANCH_NAME

# Check if a branch name is provided as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <branch-name>"
  exit 1
fi

BRANCH_NAME=$1

git checkout -b $BRANCH_NAME origin/$BRANCH_NAME
git tag archive/$BRANCH_NAME $BRANCH_NAME
git checkout master
git branch -D $BRANCH_NAME
git branch -d -r origin/$BRANCH_NAME
git push --tags
git push origin :$BRANCH_NAME
