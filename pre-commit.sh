#!/bin/bash

ROOT=./

echo "Building table of contents (J °O°)J┻━┻"

build_readme() {
  local DIR=$1
  local NESTED=$2

  local README=$DIR/README.md
  rm -f $README

  echo "# 🎵otes Table of Contents" > $README

  if [ "$NESTED" = true ]
  then
    echo "- [..](../README.md)" >> $README
  fi

  for f in $DIR*
  do
    local fname=${f#$DIR}
    if [[ -d $f ]]; then
      # Directory
      echo "- [$fname]($fname)" >> $README
      build_readme $f/ true
    elif [[ -f $f ]]; then
      # File
      if [[ "$fname" != "README.md" ]] && [[ "$fname" != "pre-commit.sh" ]]
      then
        echo "- [$fname]($fname)" >> $README
      fi
    fi
  done

  if [ "$NESTED" = false ]
  then
    echo "" >> $README
    echo "This table of contents is auto generated using [pre-commit.sh](pre-commit.sh) on every commit." >> $README
    echo "" >> $README
    echo "Install it in your 🎵otes repo using \`ln pre-commit.sh .git/hoots/pre-commit\`" >> $README
  fi

}

build_readme $ROOT false

echo "done ┬─┬/(°_°/)"

git add .
