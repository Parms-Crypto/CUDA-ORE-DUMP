CUDA add-on for ORE mining, this is bare bones. But if you know what you're doing very easy!

Bare-bones due to the fact $ORE dev has no clue what they are doing and already changing tokens to a new version.

Keep up to date since more user-friendly updates will drop when $ORE drop new version 2 code.

No I'm not ignorant and going to drop my top 1% miner advantageous full suite. I may drop the full cuda suite when my ASIC write up is complete. 
I will always be one step ahead of what I release. Maybe donations will speed things up (donation does counteract me sharing my $ORE mining secrets that drop my own rewards) 

5aqLT7MisVv9Hn9YJY1xisyVeQcr7SmDabrfZ2QoNE9q


This drop is to all the trolls in $ORE community larping as miners that don't even understand basics.

Maybe take some time to learn PoW mining and basic gametheory. Years and years of proven research by the best in the field.
Yet, you think you "know it all" and going to prove them all wrong? OK, bet!

Good luck "small miners" your leader pushed you to your own death, you now have REAL MINERS opposition against your project!

More hash = more wallets to mine without any speed losses due to computational restraints on CPU which is the weakest form of mining 

1 GH is 1 billion hashes per a second. GPU can solve 4x difficutly so far in under 3 ms on 172 instances (and counting) simultaneously with no effort.

Miner PoW history will not be broken by $ORE traders larping as miners CPU -> GPU -> FPGA -> ASIC

-SIDE-NOTE-
I have a Keccak mining pool front end suite I may drop to help push big mining farms. Maybe once ASIC come online
I'll start it up.  Might as well take a cut giving "small miners" just an oppurtunity. Mobile and phone mining will be impossible soon
and $ORE rewards will scale to 0.000000001 a hit.


GPU KECCACK ALGO HASH SPECS (hash within 1% of standard GPU miner underclocked settings)

Nvidia RTX 4070Ti - 2.454 GH/s - 280 W - 0.008 GH/W

Nvidia RTX 3090 - 2.171 GH/s - 300 W - 0.006 GH/W

Nvidia RTX 2080Ti - 1.696 GH/s - 250 W - 0.007 GH/W

Nvidia GTX 1080Ti - 1.281 GH/s- 225 W - 0.005 GH/W

Nvidia GTX 1660Ti - 0.648 GH/s -75 W - 0.008 GH/W


ASIC KECCAK ALGO HASH SPECS (in progress on these machines)

Fusionsilicon X2 - 75.0 GH/s - 980 W - 0.077 GH/W

Blackminer F1+ - 34.9 GH/s - 1000 W - 0.035 GH/W



WARNING!!! IF YOU DON'T KNOW WHAT YOU'RE DOING BURNING UP YOUR GPU IS YOUR OWN ISSUE
YOUR SUITE WILL HAVE TO ACCOUNT FOR UNDERCLOCKING AND IN DEPTH GPU MINING KNOWLEDGE, 
IM NOT HERE TO HOLD PEOPLES HANDS. 

1. Setup and Compile Cuda

https://developer.nvidia.com/cuda-downloads

* cuDNN may be advantageous additional option for some

2. Rust dependency/bindings info in cargo.toml and link.toml

3. Modify for your miner and build (establish data send/recieve with ore-cli mine.rs)

4. mine and dump all profit

5. Win!
