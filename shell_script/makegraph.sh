#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "- 引数の数が正しくありません。"
  exit 1
fi

# YAMLファイルのパス
yamlpath="configs/$1"

if [ ! -e "$yamlpath" ]; then
  echo "- 指定されたyamlファイルのパス [$yamlpath] にはファイルが存在しません"
  exit 1
fi

# lammpsの実行コマンド
COMMAND="pypy3 script/makegraph.py $yamlpath"

echo '[started Common Neighbor Analysis]'
echo -e "- COMMAND: $COMMAND \n"
eval $COMMAND
echo -e "succeeded"