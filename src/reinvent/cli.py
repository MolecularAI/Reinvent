#!/usr/bin/env python
#  coding=utf-8

import json

import click

from .running_modes.manager import Manager


@click.command()
@click.argument('file', type=click.File('r'))
def main(file):
    configuration = json.load(file)
    manager = Manager(configuration)
    manager.run()


if __name__ == "__main__":
    main()
