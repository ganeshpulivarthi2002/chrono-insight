#!/bin/bash
gunicorn main:app --preload --bind 0.0.0.0:$PORT
