--[[
  Based on pull request 54 by jroakes
]]
require('onmt.init')
require('torch')
local tds = require('tds')
local zlib = require ('zlib')
local path = require('pl.path')


local cmd = onmt.utils.ExtendedCmdLine.new('concat_embedding.lua')

cmd:text('')
cmd:text('**concat_embedding.lua**')
cmd:text('')

cmd:option('-config', '', [[Read options from this file]])

local options = {
  {'-dict_file',       '', [[Path to outputted dict file from preprocess.lua.]],
                      {valid=onmt.utils.ExtendedCmdLine.fileNullOrExists}},
  {'-global_embed',    '', [[Path to global embedding file.]], 
                      {valid=onmt.utils.ExtendedCmdLine.fileNullOrExists}},
  {'-local_embed',    '', [[Path to local embedding file. If missing, it converts the global embedding into a the t7 format]], 
                      },                      
  {'-save_data',     '', [[Output file path/label]],
                      {valid=onmt.utils.ExtendedCmdLine.nonEmpty}},
  {'-normalize',      1, [[Boolean to normalize the word vectors, or not.]], 
                      {enum={0,1}}},
  {'-report_every',   10000, [[Print stats every this many lines read from embedding file.]]}
}
cmd:setCmdLineOptions(options, 'Concatenate Embedding')

cmd:text('')
cmd:text('**Other options**')
cmd:text('')

onmt.utils.Logger.declareOpts(cmd)

local opt = cmd:parse(arg)


local function loadEmbeddings(globalEmbeddingFilename, localEmbeddingFilename, dictionary)

  --[[Converts binary to strings - Courtesy of https://github.com/rotmanmi/word2vec.torch]]
  local function readStringv2(file)
    local str = {}
    local max_w = 50

    for _ = 1, max_w do
      local char = file:readChar()

      if char == 32 or char == 10 or char == 0 then
        break
      else
        str[#str + 1] = char
      end
    end

    str = torch.CharStorage(str)
    return str:string()

  end

  -- [[Looks for cased version and then lower version of matching dictionary word.]]
  local function locateIdx(word, dict)

    local idx = nil
    idx = dict:lookup(word) 
    if idx == nil then
      idx = dict:lookup(word:lower())
    end
    return idx

  end


  -- [[Fills value for unmatched embeddings]]
  local function fillGaps(weights, loaded, dictSize, embeddingSize)

    for idx = 1, dictSize do
      if loaded[idx] == nil then
        for i=1, embeddingSize do
          weights[idx][i] = torch.normal(0, 0.9)
          loaded[idx] = true
        end
      end
    end

    return weights,loaded

  end

  -- [[Initializes OpenNMT constants.]]
  local function preloadSpecial (weights, loaded, dict, embeddingSize)

    local specials = {onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD, onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD}

    for i = 1, #specials do
      local idx = locateIdx(specials[i], dict)
      for j=1, embeddingSize do
        weights[idx][j] = torch.normal(0, 0.9)
      end
      loaded[idx] = true
    end

    return weights, loaded

  end

  -- [[Loads an embedding file.]]
  local function loadWord2vevcFile(filename, dict)
    local dictSize = dict:size()
   ---------------------------------------------------
    local loaded = tds.Hash()
    local f = torch.DiskFile(filename, "r")

    f:ascii()
    local numWords = f:readInt()
    local embeddingSize = f:readInt()
    local embeddings = {}
    local weights = torch.Tensor(dictSize, embeddingSize)

    weights, loaded = preloadSpecial (weights, loaded, dict, embeddingSize)
    f:binary()
    _G.logger:info('Processing embeddding file' .. filename)
    for i = 1, numWords do
      if i%opt.report_every == 0 then
         _G.logger:info(i .. ' embedding tokens reviewed. ' .. #loaded .. ' out of ' .. dictSize .. ' dictionary tokens matched.' )
      end
      local word = readStringv2(f)
      local wordEmbedding = f:readFloat(embeddingSize)
      wordEmbedding = torch.FloatTensor(wordEmbedding)
      local idx = locateIdx(word, dict)
      if idx ~= nil then
        weights[idx] = wordEmbedding
        embeddings[word] = idx
        loaded[idx] = true
      end

      if #loaded == dictSize then
        _G.logger:info('Quitting early. All ' .. dictSize .. ' dictionary tokens matched.')
        break
      end
    -- End File loop
    end
    f:close()
    if #loaded ~= dictSize then
      _G.logger:info('Embedding file fully processed. ' .. #loaded .. ' out of ' .. dictSize .. ' dictionary tokens matched.')
      weights, loaded = fillGaps(weights, loaded, dictSize, embeddingSize)
      _G.logger:info('A total of '.. dictSize-#loaded  .. ' words have been randomly assigned according to normal distribution')
    end
    return weights, embeddingSize
  end

  local function normalize(weights)
    for idx = 1, weights:size(1) do
      local wordEmbedding = weights[idx]
      local norm = torch.norm(wordEmbedding, 2)
      if norm ~= 0 then
        wordEmbedding:div(norm)
      end
      weights[idx] = wordEmbedding
    end
    return weights
  end

  local function loadWord2vec(global_filename, local_filename, dict)
    local dictSize = dict:size()
    _G.logger:info("Loading global embeddings")
    global_weights, global_embeddingSize = loadWord2vevcFile(global_filename, dict)
    _G.logger:info("Loading local embedding")
    local_weights, local_embeddingSize = loadWord2vevcFile(local_filename, dict)
    ---------------------------------------------------
    _G.logger:info('Merging the Global and Local embeddings...')
    local embeddingSize = global_embeddingSize + local_embeddingSize
    local loaded = tds.Hash()
    local weights = torch.Tensor(dictSize, embeddingSize)
    weights, loaded = preloadSpecial (weights, loaded, dict, embeddingSize)

    _G.logger:info('Done')
    return weights, embeddingSize
  end

  if localEmbeddingFilename ~= '' then
    local weights, embeddingSize = loadWord2vec(globalEmbeddingFilename, localEmbeddingFilename, dictionary)
    if opt.normalize == 1 then
      weights = normalize(weights)
    end
    return weights, embeddingSize
  else
    local weights, embeddingSize = loadWord2vevcFile(globalEmbeddingFilename, dictionary)    
    if opt.normalize == 1 then
      weights = normalize(weights)
    end    
    return weights, embeddingSize
  end
  
end


local function main()

  _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level)
  local timer = torch.Timer()
  
  local dict = onmt.utils.Dict.new(opt.dict_file)

  local globalEmbedFile = opt.global_embed
  
  local localEmbedFile = opt.local_embed
  if localEmbedFile ~= '' then
    assert(path.exists(localEmbedFile), 'embeddings file \'' .. opt.local_embed .. '\' does not exist.')
  end
  
  local weights, embeddingSize = loadEmbeddings(globalEmbedFile, localEmbedFile, dict)  
  
  _G.logger:info('saving weights: ' .. opt.save_data .. '-embeddings-' .. tostring(embeddingSize) .. '.t7' )
  torch.save(opt.save_data .. '-embeddings-' .. tostring(embeddingSize) .. '.t7', weights)

  _G.logger:info(string.format('completed in %0.3f seconds. ',timer:time().real) .. ' embedding vector size is: ' .. embeddingSize )

end

main()
