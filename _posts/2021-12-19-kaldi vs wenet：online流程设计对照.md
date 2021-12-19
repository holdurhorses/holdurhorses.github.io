---
title: kaldi vs wenet：online流程设计对照
author: Zhang
date: 2021-12-19 21:10:00 +0800
categories: [ASR]
tags: [writing]
render_with_liquid: false　
---

　　ASR online实现是工程实现必不可少的一环，可以随着用户实时语音流的输入及时返回识别结果，对于用户体验感非常重要。kaldi/wenet都支持全流程的online实现，主要包括online feature extractor、online model inference、online decoding。由于wenet借鉴了kaldi的不少实现思路和代码，所以本文以kaldi为主介绍online实现的整体思路，并在差异化环节分别介绍kaldi和wenet的实现和简单原理，本文还是遵循kaldi的初衷使用“online”(而非“流式”或者“实时”)作为一种谨慎描述。

# Take care

完成一个Online系统，要妥善处理几方面：

- **processing level**：一般有三个level，分别为frame-level，chunk-level，utterence-level。hmm-based asr由于遵循一阶马尔可夫假设，基本可以做到frame-level。self-attention based在ASR（短时上下文依赖特性）应用中可以用chunk-level，当然utterence-level会更好。同时在最上层一般都是utterence-level综合决策，如voice activity detection,end-point detection or rescoring 。
- **offset**：全局的偏移量。如kaldi decoder中num_frame_decoded_，wenet中transformer positon encoding。
- **cache**：由于网络的上下文依赖，feature一般需要cache多帧，如kaldi中mfcc或者wenet中的fbank的queue实现。同时网络中间层的输出由于存在复用性，也会进行cache。decode部分同样依赖历史解码信息，在kaldi中的token保存，transformer中decoder层的自回归解码。
- **interact**：不同模块之间的交互逻辑。如feature extractor和model inference模块之间的交互。model在online模式下是如何使用feature的。

对于上述细节在kaldi和wenet中再着重展开一下。

# feature extraction

　　wenet基本采用了kaldi中FeaturePipeline的接口和实现逻辑，但是又略有差别。

| 目的和差别                                                   | kaldi                                                        | wenet                                                        |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **waveform输入**，kaldi中实现了resampling<br />操作，wenet做了简化 | void AcceptWaveform(<br />BaseFloat sampling_rate,  <br />constVectorBase<BaseFloat> &waveform) = 0; | void FeaturePipeline::AcceptWaveform(const <br />std::vector<float>& wav); |
| **下层访问接口**，kaldi中提供了随机访问的接<br />口，使用NumFramesReady()信号，而<br />wenet严格按照顺序方式生产和消耗。同时<br />当模型需要的特征得不到满足时，会阻塞在<br />ReadOne() | void GetFrame(int32 frame, <br />VectorBase<BaseFloat> *feat); | bool ReadOne(std::vector<float>* feat)                       |
| **cache**，kaldi中采用了简单的双端队列实<br />现，而wenet考虑到多线程部署问题，采<br />用了更为智能的BlockingQueue，但会阻<br />塞线程。 | std::deque<Vector<BaseFloat>*> items_;                       | BlockingQueue<std::vector<float >> feature_queue_;           |
| **offset**                                                   | feature_.Size()                                              | num_frames_                                                  |
| kaldi**中其他有意思的实现**                                  | 惰性计算，如ivector                                          |                                                              |



# model structure and inference

　　由于kaldi和wenet采用了完全不同的网络结构，所以该处的实现区别很大，TDNN实现相对简单，而conformer/transformer的实现复杂一些。kaldi将model inference整合成单独的模块，而wenet采用了torch jit script进行模型前向并整合到torch_asr_decoder中。

| 目的和差别                                                   | kaldi                                                        | wenet                                                        |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **feature输入**                                              | 非阻塞式，NumFramesReady()                                   | 阻塞式                                                       |
| **下层访问接口**，kaldi中提供了<br />LogLikelihood()惰性触发网络<br />计算，而wenet每个chunk直<br />接触发网络前向计算。 | BaseFloat LogLikelihood(int32 frame, <br />int32 index);     | model_->torch_model()                                      ->run_method("ctc_activation", chunk_out) .toTensor()[0]; |
| **cache**，kaldi中所有的cache内部<br />进行统一管理(内部相对复杂的<br />dilated 1d-cnn cache结构)，<br />只要满足一个chunk的特征数据<br />即可计算。wenet显式传入cache，<br />同时left context feature也需要cache。 | 所有cache内部管理。                                          | **subsampling_cache:** subsampling的输出的cache。<br />**elayers_output_cache**:所有conformer block的输出的cache。<br />**conformer_cnn_cache**: conformer block内部 causal conv层的左侧依赖的输入cache。 |
| **网络结构和上下文**，网络基本没有<br />太多共同之处。在subsampling上<br />略有共同点，kaldi 实现了1d-cnn<br />的叠加，wenet使用2d-cnn，优化<br />帧率和解码计算量。kaldi为了避免<br />右侧依赖过长而使用了不对称的网络结构<br />，使右侧依赖相对较小。wenet只<br />依赖左侧chunk和conformer 内部<br />使用左侧依赖的casual cnn，使得<br />只有在subsampling时引入少许右侧依赖。 | subsampling: dilated 1d-cnn<br />叠加降低计算量和跳帧解码降低帧率<br /><br />**一种可能配置**<br />chunk_size =18<br />right_context = 9<br />left_context = 12 | subsampling: 2d-cnn直接降低帧率且降低计算量<br />**一种可能配置**<br />chunk_size =16\*4(wenet直接从decoder角度描述chunk_size，则实际的net chunk_size = subsampling_factor\*chunk_size)<br />right_context = 6<br />left_context = num_left_chunks\*chunk_size |

# decoder

　　kaldi一般采用基于WFST结构的HCLG解码图进行解码，而wenet由于端对端网络结构的特殊性，采用了多种解码策略，比较重要的有两种：1.基于WFST结构的TLG解码图进行解码。2.基于CTC prefix beam search的direct decode和attention parallel rescoring。方法1和kaldi基本一致，而方法2和kaldi无关。同时两个decoder的含义略有不同，kaldi decoder只负责给定网络前向结果后的解码搜索。而wenet将网络前向和解码搜索整合到torch_asr_decoder中简化了流程，但是实际的解码在SearchInterface的派生类中实现，当实现为CtcWfstBeamSearch和kaldi几乎一致。
　　同时由于基于WFST的解码方式基本依赖左侧，因此具有天然的online特性。对于lattice-faster-online-decoder，相对去之前lattice-faster-decoder实现了更为高效的get partial result，cache也只是为了记录active_tokens_。

| 目的和差别          | kaldi                                                   | wenet                                                        |
| ------------------- | ------------------------------------------------------- | ------------------------------------------------------------ |
| **score获取**       | BaseFloat LogLikelihood(int32 frame, <br />int32 index) | ctc_log_probs = model_->torch_model()>run_method("ctc_activation", chunk_out).toTensor()[0];<br />实际由于复用了kaldi的LatticeFasterDecoder，所以当search函数采用kWfstBeamSearch模式时，采用了fake的decodable对象进行接口转换。 |
| **decoder**         | lattice-faster-online-decoder                           | lattice-faster-online-decoder                                |
| **Graph结构**       | HCLG                                                    | 采用CtcWfstBeamSearch时会在TLG上进行搜索。 采用CtcPrefixBeamSearch时则没有使用WFST。 |
| **search strategy** | beam search, max_active, min_active                     | CtcPrefixBeamSearch: two pass beam prune<br /><br />CtcWfstBeamSearch: kaldi beam search, blank_skip_thresh skip blank frame. |
| **output**          | one-best, n-best, lattice                               | CtcWfstBeamSearch: one-best,n-best,lattice<br />CtcPrefixBeamSearch: one-best,n-best |

# reference

[1] https://github.com/kaldi-asr/kaldi

[2] https://github.com/wenet-e2e/wenet

[3] [Wenet网络设计与实现4-Cache - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/396703996)
