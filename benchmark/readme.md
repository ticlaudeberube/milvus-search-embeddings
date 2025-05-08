## cpu opritmization benchmarks
## Hugging Face
    - get a token form Hugging Face
    - Insert token and activate set-hf-token.sh script   
## Ollama 
### Get physical CPUs
```
 sysctl -n hw.physicalcpu   

```

### Thread check
Use script ollama-threads-check.py to test threads

### Set Ollam threads
    - Use set-ollama-threds.sh