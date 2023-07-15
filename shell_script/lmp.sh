#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "- 引数の数が正しくありません。"
  exit 1
fi

# YAMLファイルのパス
yamlpath="configs/$1"

if [ ! -e "$yamlpath" ]; then
  echo "- 指定されたパス [$yamlpath] にはyalmファイルが存在しません"
  exit 1
fi

# yqを使用してYAMLファイルを解析し、設定を読み込む
lmpfile_name=$(yq eval '.lmpfile_name' "$yamlpath")
dirname="lmpfile/$(yq eval '.dirname' "$yamlpath")"
lmppath="$dirname/$lmpfile_name"

if [ ! -e "$lmppath" ]; then
  echo "- 指定されたパス [$lmppath] にはlammps実行ファイルが存在しません"
  exit 1
fi

# lammpsの実行コマンド
COMMAND="cd $dirname && lmp -sf gpu -in $lmpfile_name && cd -"

echo '[started the MD simulation]'
echo -e "- COMMAND: $COMMAND \n"
eval $COMMAND
echo -e "succeeded"