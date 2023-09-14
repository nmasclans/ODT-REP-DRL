#!/bin/bash

shopt -s extglob

rm -vrf !("README.md"|"user_config"|"clean_this_dir.sh")
