# config file for processing data

- copy `defalut.json` and adjust for each model

```json
{
  "FT_ANALYSIS": true,
  "FT_ANALYSIS_SIZE": {
    "start": 10,
    "end": 50,
    "step": 10
  },
  "FT_TIMESTAMPS": 50,
  "FT_DIRECTION": 50,
  "FT_PACKET_SIZE": 50,
  "FT_BURST_DIR": 50,
  "FT_BURST_SIZE": 50,
  "FT_CUMULATIVE_DIR": 50,
  "FT_CUMULATIVE_SIZE": 50
}
```

# parameters

### categorical features

- `FT_ANALYSIS`: true/false

  - whether to contain following features
  - incoming count/rate, outgoing count/rate에 대한 중간 집계 시점을 결정
  - 주어진 data 상에서 배열의 최소 길이는 50이지만, 만약 50보다 큰 값을 넣는 경우 모자란 길이만큼 뒤에 0으로 패딩처리 됨
  - 즉, [1, 2, 3, 4, 5, 6, ..., 50]을 다 넣는 경우 50번째 패킷까지 누적값을 계속 집계

  > `count`: total number of packets
  > `incoming_count`: total number of incoming packets
  > `incoming_rate`: incoming_count / count
  > `outgoing_count`: total number of outgoing packets
  > `outgoing_rate`: outgoing_count / count

- `FT_ANALYSIS_SIZE`: list

  - list of checkpoint to analyze

### sequential features

feature의 몇번째 원소까지 데이터에 사용할 지 결정함

- `FT_TIMESTAMPS`: int
- `FT_DIRECTION`: int
- `FT_PACKET_SIZE`: int
- `FT_BURST_DIR`: int
- `FT_BURST_SIZE`: int
- `FT_CUMULATIVE_DIR`: int
- `FT_CUMULATIVE_SIZE`: int
