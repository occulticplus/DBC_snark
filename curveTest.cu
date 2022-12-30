#include<stdio.h>
#include<sys/time.h>
#include<random>
#include <cstring>
#include <vector>
#include "multiexp/gpuec256.cu"
#include "multiexp/cuda_common.h"
#include "multiexp/uint288.cu"
#include "multiexp/constants.cu"

// libsnark header
#include <cassert>
#include <cstdio>
#include <fstream>
#include <libff/common/rng.hpp>
#include <libff/common/profiling.hpp>
#include <libff/common/utils.hpp>
#include <libsnark/serialization.hpp>
#include <libff/algebra/curves/mnt753/mnt4753/mnt4753_pp.hpp>
#include <libff/algebra/curves/bn128/bn128_pp.hpp>
#include <libff/algebra/curves/mnt753/mnt6753/mnt6753_pp.hpp>
#include <omp.h>
#include <libff/algebra/scalar_multiplication/multiexp.hpp>
#include <libsnark/knowledge_commitment/kc_multiexp.hpp>
#include <libsnark/knowledge_commitment/knowledge_commitment.hpp>
#include <libsnark/reductions/r1cs_to_qap/r1cs_to_qap.hpp>

#include <libsnark/zk_proof_systems/ppzksnark/r1cs_gg_ppzksnark/r1cs_gg_ppzksnark.hpp>

#include <libfqfft/evaluation_domain/domains/basic_radix2_domain.hpp>


using namespace std;
using namespace libff;
using namespace libsnark;

#define CONST_SCALE 14
const int N_BIGNUM = (1<<CONST_SCALE) + 1; // maxsize = 2^15, for not run out of memory.
const int BATCHSIZE = (1 << CONST_SCALE) + 1; // 
const int N_POINT = BATCHSIZE;
#define CONST_THREAD_PER_BLOCK 32
const int N_THREAD_PER_BLOCK = CONST_THREAD_PER_BLOCK;
const int N_BLOCK = ((BATCHSIZE+N_THREAD_PER_BLOCK-1)/N_THREAD_PER_BLOCK);
#define CONST_BIGNUM ((1 << CONST_SCALE) + 1)
#define CONST_BATCHSIZE CONST_BIGNUM
#define CONST_BLOCK ((CONST_BATCHSIZE + CONST_THREAD_PER_BLOCK - 1) / CONST_THREAD_PER_BLOCK)

const UINT64 h_R2[4]={0x000007a2000e90a1L,0x1L,0x0L,0x0L};

const UINT64 h_Gx[4]={0x59F2815B16F81798L,0x029BFCDB2DCE28D9L,0x55A06295CE870B07L,0x79BE667EF9DCBBACL};
const UINT64 h_Gy[4]={0x9C47D08FFB10D4B8L,0xFD17B448A6855419L,0x5DA4FBFC0E1108A8L,0x483ADA7726A3C465L};

const UINT64 testGx[4]={0x252931db128244c9L,0x80ec2e92027d7e6eL,0x32c5ee6d51cb1e89L,0xb89bd74c7352f570L};
const UINT64 testGy[4]={0xd8cbce4f20d0d9e4L,0x5b636389add7cc6eL,0xccd07463f61e7fbeL,0x13fae72c0d3c849bL};
const UINT64 testGz[4]={0xbc5c645f1b1c297dL,0x0ba1469cd0bdd88aL,0x40bad30e143dcdceL,0x4bba49beb75cce43L};

Jpoint testG = {{0x252931db128244c9L,0x80ec2e92027d7e6eL,0x32c5ee6d51cb1e89L,0xb89bd74c7352f570L}, {0xd8cbce4f20d0d9e4L,0x5b636389add7cc6eL,0xccd07463f61e7fbeL,0x13fae72c0d3c849bL}, {0xbc5c645f1b1c297dL,0x0ba1469cd0bdd88aL,0x40bad30e143dcdceL,0x4bba49beb75cce43L}};


//#define PRECOMPUTE // precompute test.
#define CUDA_VISIBLE_DEVICES 1

// 32 dbcs for test. to long dbc are very slow to compile.
const int SMALL_DBC_NUM = 32;

unsigned long long px[] = {0xc1291ea73f54ce04ll, 0xbaae63f186512829ll, 0x849ffc311521657all, 0xdb05b8be3bc3ea12ll, 0xec734eb90e60d146ll, 0x47e3a925f4c2f8d3ll, 0xe404a3c34bfc4331ll, 0xa09f73b213ee39abll, 0xdb13ac478ea2128ell, 0x2c53e4e8f64e797dll, 0x39b7a582468b2b26ll, 0x8a7cc04c32f86ce1ll, 0xcd358ec4bf05d09ell, 0x6b2044c68221ce4ll, 0x87a85903221d0748ll, 0x46f323835084b5a9ll, 0x46f97386edeaf803ll, 0xa49b5c87eb315db7ll, 0xe3081f1ad7d6d04all, 0x83059f0ea01a4aeell, 0xee6fba127fec7397ll, 0xaba6a4a20bad9c85ll, 0x7644585b6c958d2cll, 0x84938de81702b2a3ll, 0x1a3ad24da633218ell, 0xec8c6b7a2e2d7fd4ll, 0x5893289ae2b2d9bll, 0xe5ef285c943653aell, 0xa04bd5faef99b5cell, 0x978380dff860c313ll, 0xa993bd51570b133ell, 0xb6d827a6ffe8c506ll, 0xf3fc602ee2294187ll, 0x885972979b942caell, 0xfe36bc732ebf4a28ll, 0x44524be479e97992ll, 0xf46d14dc6fdec9fll, 0x8548cb4caa31b9d7ll, 0xeefb2bfedba2f870ll, 0xee2cf31377c30c02ll, 0x8abcb93f4f13ae56ll, 0x4b03bbfee680553ell, 0x17a359c48f5f772ell, 0x4b8590bc7ad10993ll, 0xd81ef93a1c4bc230ll, 0x3fe5e94a6137a639ll, 0xdf6937fecfe2ad57ll, 0x70c0b2bac6f886f1ll, 0x9f04236a79cf01f6ll, 0x13e1640167007df8ll, 0xf780f2e0d9a1ae0fll, 0xbf7d094c158291cfll, 0x212b875aa42d2b3cll, 0x446ebb199ab8c39fll, 0xdcaf114ec3d14cfdll, 0x2ba87b8b8d3f1f59ll, 0x9bcd364ea3325d90ll, 0x3a47a15979d76791ll, 0x587c8cc21bc825dell, 0x89c87f72dfc370fcll, 0xb0d283bbfa0e29b3ll, 0x26308aa1138b7509ll, 0xaf529fa55238314ll, 0x39816c256e503a7ell, 0x5f22c397ef9fe0all, 0xb9e5bdb683e15a95ll, 0xaaa3fb7ade2d4d88ll, 0x4d7e7ee5951e2398ll, 0x5470acdb8695857ll, 0x5166d1ad1ad03d79ll, 0xf3305c87d6eb6a66ll, 0xc176900a101104fll, 0xce06d34209f8948all, 0xd82193ffdfe2c9e5ll, 0x931075c79ce0d306ll, 0x775fb75a7642d179ll, 0x888cda266c2aad54ll, 0x40fc82795775f1fall, 0x497bad018e821b36ll, 0xa29b1ff55c8448c9ll, 0x69af3b5510579da2ll, 0x2f1fbea9ba3d7984ll, 0x774fd33a1752595cll, 0x56a3a84b5041c8ebll, 0x7cde84ea0461ba1cll, 0x603181bf3283c6d6ll, 0x62f2c535e53dcee8ll, 0x2451e9cc1fdf3d7ll, 0xda232c0afcdc6597ll, 0xeb4ba40875b6a4dall, 0x2d5f9584192f072bll, 0x3198177f0b01bcdcll, 0xacd3d97303bbd12cll, 0x8db5b517cbe942e0ll, 0x51d09eb6d2933d2ll, 0xe6d9c03b36990772ll, 0xb964889dbf78ebd5ll, 0xb9811d6fc0356d18ll, 0xa2410b8e143e5130ll, 0xcc38ce20d9807de7ll};

unsigned long long py[] = {0x8ceceeb28411ba32ll, 0x3e781792a382e71ell, 0xd3fa02dae181fc95ll, 0x2e304597764f60a6ll, 0xfce1486fcc18d692ll, 0x1375c5b6e87f2c3all, 0xb48702fc8a96ba27ll, 0x57e69125d0114f2cll, 0x362d1b3393fa7588ll, 0x78796df78872500dll, 0x9138dfc1bdc15a3dll, 0xa167f3d2414d0205ll, 0x1b9b9a5f34b8afecll, 0x583a0f67a99a3788ll, 0xc4acd28dffe1410cll, 0xf07e21417abbcc3cll, 0x3ade3c45dfbc863fll, 0x9eda6f139f31b018ll, 0xe0adb41adfca8622ll, 0x15ea15417084f16bll, 0xb3b2c55417193756ll, 0x13712c30fdc9e718ll, 0x2952caab2f05eab1ll, 0x358afc9cbd7e5192ll, 0xd536b8e7f4c507d6ll, 0x66c5a5ff171507e0ll, 0x725b2f3fb86bac7bll, 0x9e36795fd946b087ll, 0x357fcb8b68e5b458ll, 0x7766f17d2f3054d7ll, 0x8343812a93422063ll, 0xa1f100198fe5c71dll, 0x68620cb7296393a2ll, 0xd95ea31298a2c21dll, 0x77d391aa12784ca4ll, 0x8fcf1ba16eaf7c5bll, 0xc68cfe2925c1ac51ll, 0x2352bcdd153d436cll, 0xcfeeeb13fb787307ll, 0xac675e68a3bff56bll, 0xd621070cf7495c20ll, 0x31c021898ac69039ll, 0x8f0c823687669c4all, 0x144b3504c3329402ll, 0x10a0a577850a8ac9ll, 0x2a0026f19fabf26ll, 0x8b726c9485375803ll, 0xf2e492176f575e7ell, 0xbba9a542f2b6cf79ll, 0x117cb4563c3d2f96ll, 0x65b88205addecb52ll, 0x615c0a66a7d6a4dell, 0x35f4afe2701217e2ll, 0x15e4a785245be7e3ll, 0x2993bd9756922e8ell, 0x5c2ff49d4f54fda0ll, 0xa82c16071118f14fll, 0x800ca14fc42286e7ll, 0x1e5b6da6f7f886e3ll, 0x37997aecc99fc704ll, 0x6991710fc91f47e3ll, 0x9877646ac9334869ll, 0x61ac8c5c317aba27ll, 0x69b0ede62479a906ll, 0xf1d3bf52ace992d5ll, 0x1c9980570a29208all, 0x53784402059629d6ll, 0x9698f219e1c2443all, 0xf6f31dc5852bdbbbll, 0x261e0d1867a702f8ll, 0xd166e64d5d8829b9ll, 0xc77dd21b6cbdd1b6ll, 0xe69accceb5e2b4e9ll, 0x69baa5bbe515be9dll, 0xb616554bece011cdll, 0xa34be20aa7d9c90cll, 0x5e7cc1b8428e709cll, 0x5c58a6ac2c829e11ll, 0x9a3dce73b43a903ll, 0xa58d08ec4c58bd52ll, 0x7f7751124655e13ll, 0x5424aa4fdbf8b18dll, 0xc982cbae45ccce95ll, 0x9398e8c27258250dll, 0xf17129bfa5b1b510ll, 0x41fe283dfb4327afll, 0x5f2f33e7714c97dall, 0x559cf7242bb8fe8ell, 0x17b3a3bb46dd3e91ll, 0x1af7d0a9f1b04817ll, 0x6d23e0dd841a2672ll, 0x8f6d1a2693629455ll, 0xff20d2520292748dll, 0x24d2beca6756a36dll, 0x7644c94f54b6cb43ll, 0x8b5d5592833e959fll, 0xb0c016d4371e5f9ell, 0xa0acac44e0ca9561ll, 0x9d27f8c2c840e7ell, 0x73b4245bfb56fc89ll};

unsigned long long px3[] = {0x4d3c296b3e8b6988ll, 0xddcc9ca08396d1cbll, 0x7ad77e63364511dfll, 0xa0ac877d59fe43all, 0x4a87562b177eae72ll, 0xf52831b49c6a6112ll, 0xf0b0a46021a488d2ll, 0xd4575222d4cda88bll, 0x9fd2acb1e587292bll, 0xd8c007515b734f4ll, 0x86d785c545c96c0dll, 0x81b2826c2c4d2dd8ll, 0x94001024d35f4e03ll, 0xa7f67326231808c0ll, 0x49e61768c043f864ll, 0xa6bb07c8044cb754ll, 0x3bc5fce63b5c6eb7ll, 0xe975105a35c835b0ll, 0x5449ac7731bf26a6ll, 0x8bb21922e7870e9cll, 0xbf51fc11ed6b7becll, 0x90c2a7dd970b6a7all, 0x87b1d7c096ab0fd1ll, 0xef4efa6172ed7f00ll, 0x70bfa5991f8040b5ll, 0xf63d2f69ab92e571ll, 0x30f4f32599299452ll, 0xc55ac8e19c98b574ll, 0x1e7bd7e1f2a7f00all, 0x9bcc5a675c011ab3ll, 0x1b0ac2245d3f3e60ll, 0x6f91233e31a13e1cll, 0x15c3bbf222eb03b3ll, 0x39c2af9d552948efll, 0x51a6d8ded9469e3bll, 0x162d2cf1cfd90c8ell, 0xc3241eee21e04335ll, 0x1f7ef0ec35bb3a16ll, 0x2082ddcd712cea20ll, 0xea6dd878e1cf83bell, 0xe20708512d57af51ll, 0x53d8c5e73bfd68e7ll, 0xbc06c96708a2d80dll, 0xe5303f14e2c4ed2ll, 0xf2baa21ec43fbf3cll, 0xc66aaf40605089d8ll, 0xb98898d342d1a70all, 0x79eaa5df944cdcc3ll, 0xa4ae33d8a9392101ll, 0xd2bd0d1e685b2895ll, 0xf5dee131682af06all, 0xee9d971db32100a9ll, 0x979d6bc8cba8f21all, 0x627a73d284d1e077ll, 0x3791b8bed2f990ecll, 0xf1b664cf858bf41ll, 0x8bf00b84d5ef3043ll, 0x5290e9b8024f8d15ll, 0x586a32e199bc60afll, 0xb0011ca2bfc88320ll, 0xd246570ad39d881dll, 0xa8d58a8fb5195e07ll, 0x2d81c5167f40a226ll, 0x4d5f609d233f8b61ll, 0xe9c3989a896b89d7ll, 0xeb4f7588bf903347ll, 0x833013f909b93343ll, 0x9a573acb50f4a9ebll, 0x66d01a5810f8d137ll, 0xaa9f929069bcc2dell, 0xa6adb6c784046febll, 0x773fb2dd1fca11d2ll, 0xf8504ef7bbb60baell, 0xb8ff3fd004c40d6dll, 0xa2709481fa5be1f4ll, 0xe3b67f87945b1f0dll, 0xb8c2b550fe024263ll, 0x162da9f5da318c7ell, 0x43bd85c11f0e6f40ll, 0x576e83352bbff70bll, 0x6f532e35310c746ll, 0xde275c4ae8b7c108ll, 0x366542ca116d445dll, 0x192377a9f89ad8f5ll, 0xee8f3b67d22d5d99ll, 0x579cb13009210d66ll, 0xf3873431e3850df3ll, 0x7c69e10852936f22ll, 0x2b3a9bd371dfaba7ll, 0x6be9676d1bbce350ll, 0x7e6ee26c8b0dc356ll, 0x1b3261a0e40f0779ll, 0xa9706717ed456ab8ll, 0x5d765dabbeb674c6ll, 0x7d739283e4f43033ll, 0x8e305790a447736all, 0xe352f5943cf487fbll, 0x64398b5b8cbe1143ll, 0xfc2cb7207d937bd2ll, 0x7753cf89671c8cdell};

unsigned long long py3[] = {0xcacdf749cb8d5f19ll, 0x3a26b17a6f13f006ll, 0x1cde4777eaf1fa11ll, 0xa4414ed881c0e56dll, 0x3b571c5fa5508c0ell, 0x7cdc1849f76f2ec2ll, 0x1a468db56c8568e2ll, 0x8c57ec98bf38fe59ll, 0xd10b947bbcaf4bf5ll, 0x4e64eb670fae53bell, 0x430c61252c4de283ll, 0xab68117b7ff30299ll, 0x26a8f75f7d79ee30ll, 0x1053de0301a4bccbll, 0xbb50c1849212a0f1ll, 0x423862d46548c4bell, 0xc31f86616d9390ecll, 0xfa83bcfd2bc8718fll, 0x22d6538e7ca2b142ll, 0x21ebff421deb94b7ll, 0x8a5128224b2e1e5bll, 0xb518a254c60bd352ll, 0x8c10a7ec1dd632cfll, 0xb8b94fd1a20f561ell, 0xcbf7274e4320fde9ll, 0x7f825189b430ad19ll, 0x7fc5b0de13e6ae42ll, 0xb61ac93784c46e70ll, 0x86e902c618d1d0e2ll, 0xc7ab820d0f5ddfa6ll, 0x2927e90be13fcc69ll, 0x5a7a8539a069fa70ll, 0x8fa0db2c14690950ll, 0xee3365a358ca1d1bll, 0xd2107bae0e0d2c93ll, 0xb19305443f66057ll, 0x505575ab92a3dbd0ll, 0xfc0d0110bab28c74ll, 0x64b5014ed56a5165ll, 0x210a5c2d815d6c7ell, 0x71c2a24bc399717ll, 0x2f7c82a1be3ed40all, 0xa0fe03461542dc9all, 0x9c8175bc3b7aac92ll, 0x52a5d275fa70f38fll, 0x7f280e75ef818498ll, 0x782e3358a54ea194ll, 0xb286ede5b6c69b77ll, 0xb2fae70f5b4cb3e6ll, 0x985669303148fc90ll, 0xba4f880f241bfd5ll, 0x5134a1168ffb9c2bll, 0x88aaa0f440c2fc84ll, 0x732a815a26af6096ll, 0x8ada0e8b66cc7c48ll, 0x85d80c3e63934b97ll, 0xc276948f76dc160cll, 0x83354992d9f8f4ecll, 0xdb89f215877d3fd2ll, 0x607cf4a51e6d4beall, 0x7bc812a269cdb273ll, 0x5e91c14838a6da32ll, 0xeea1beca0c82783cll, 0xb402c8733b869ce8ll, 0x6bb6539a3a626f9fll, 0x65c3c28b9fb0dd7ell, 0xfd31658429a1fd8cll, 0x283fb6056d1dbd54ll, 0x4d1fd3ab9ce96b92ll, 0x59307fca2f3e66d2ll, 0x47899b04717f8830ll, 0x2c04c9c438a97ccell, 0x41a43e034990ccb2ll, 0xac5c62a314ae7fc1ll, 0x2bc9c15f5ce5f21dll, 0x4351c7f4bc543587ll, 0xe714a0633eb54391ll, 0x49a981d8f96d8c83ll, 0xcc039db192259e41ll, 0xb0365ab40ef69a54ll, 0xf7c0aa189d06d5a4ll, 0xe3c9ef6dbcbe0cb3ll, 0x5053f2e2d58472e7ll, 0xeed3a254c43ae758ll, 0x783efea48bc01e51ll, 0xfd450216e34bb240ll, 0xaa59875b491dfe0dll, 0x18c4e66c88aa43a4ll, 0x21ea3d478bc5b59bll, 0x50b61cc4607a4882ll, 0x7c3a24521c8e8bb8ll, 0xdcc23b4a6fae5af9ll, 0x34f723ffe88c492ll, 0x7a1049306af2c39cll, 0xb8fafb10e9e11d70ll, 0x45e280142165ae6ell, 0xd1bc897d2caf1fb5ll, 0x58c41c04b69ddcfdll, 0x3869b0569cc204c5ll, 0xfc0be1c06a7ecb4ell};


unsigned long long dx[] = {0x3e9b58b481a0c1e7ll, 0xd2790f58ad702ac3ll, 0xf3df624ee861f1all, 0x7422dbc25514e317ll, 0xb6c363056b6d570ll, 0x64b8ed462321d1b1ll, 0xfaf242de5b1e7585ll, 0xd4c307f48c33830fll, 0xd418348214808c65ll, 0xf8932fa155f832ebll, 0xd9145a22f78c225cll, 0xac037fc3f8423648ll, 0xa28bafa272ad91a5ll, 0x9795cc3f52dd1fdfll, 0x617bf6ea64012f39ll, 0x15c959cd69b96173ll, 0x79938dacb05340f0ll, 0xb73c17845e5f0655ll, 0x3caed57f4bd41d80ll, 0xa1ce5385c6053c48ll, 0x2970d3e74b38c8a2ll, 0x981e9824714f72f9ll, 0x8d5deba7fe2831cfll, 0x94fc5ff28dd55157ll, 0x9b7d882b71f43546ll, 0x7c6c74ebad9a3f32ll, 0xbebd6b634e1a88eell, 0xeb0ac31cf931b374ll, 0x7861226eaee648bll, 0x3862d67ed71d09f5ll, 0x1505d4e24a270420ll, 0xf8b0088e5648acecll, 0x90c1545404bc4a0ll, 0xe297c7d92264f5fbll, 0x9ab771c2882e07a5ll, 0x30a271fd9414e288ll, 0xc47f35f595e118b5ll, 0x552c5a97b9aa7516ll, 0x144f974df1b2ad09ll, 0x427e96ea27f5846bll, 0x78820a6c772f2688ll, 0x47a88dfd3542f024ll, 0x40c3d261f513e1a8ll, 0xd62b6192ae7bdab9ll, 0x77f9db302dcd5c28ll, 0xba5881b5c7ddd140ll, 0xfe44a6e288bf3851ll, 0x4d1f56e7962c2dell, 0x3c0ca396a6f16a01ll, 0xdea3647470db11b2ll, 0xb1dfb8febcc7541bll, 0x91d5f4ac42502ad1ll, 0x43a3f5dc4363be16ll, 0x60a7186206c08f8bll, 0x4ad258e55da97160ll, 0x510df6e5c3232ec7ll, 0xa2da54efe96f9514ll, 0xe3c1b95dad759e1bll, 0xd03e5645dbea5195ll, 0x2b64bdbb8c1cd0a6ll, 0x3d64efeb284b8ad2ll, 0xed645de57f9311ccll, 0xa9caeb58742da602ll, 0xabb9772059774479ll, 0xa23cae275137c4d1ll, 0x826d0433243e9a9ell, 0x95eba298e227b592ll, 0x3a2ed739fb98c7e9ll, 0x5cfebb35d7f90e1bll, 0xf732aff4ff5a4fb7ll, 0x748112df101e87fcll, 0x834fdc11baab3838ll, 0xd75c9e46d0aee654ll, 0x1745092a6abf8770ll, 0x1362d90dcded199ell, 0x863dfc5e91a7a9b5ll, 0xd18f3cd686e412fll, 0xe04fe216bcc01378ll, 0xb3a1cd3bdce4c117ll, 0x2283efec24582825ll, 0xf744c2c74567e2f5ll, 0x97afc09b7b282e43ll, 0xd198c777f05886bbll, 0x6b23e4c0f51dadb5ll, 0xdae050c76c1eddb3ll, 0xdaab67905a7d0659ll, 0x2b174248a0584645ll, 0xd34e1a4d0bd960b1ll, 0x808d3cc6dd83fc59ll, 0xbf91a51648dd0960ll, 0xf38567e92f48df68ll, 0xacb571f8dd66500ll, 0xecd41f0b86138682ll, 0x21e873127c645bb6ll, 0xb5a0e535ee29b6b1ll, 0xeaeeab56d06c2c0ell, 0x929db043d8f2cdfall, 0xc0d2a2608be97e5ll, 0x426dc138552c4432ll, 0x868a971759b755ccll};

unsigned long long dy[] = {0x3542486d3e8384e3ll, 0x13d051ad6b464dedll, 0x5b42fc37a6293dbdll, 0x6e4b2cbd7e92276fll, 0x9fef0e9c8b15264fll, 0xabb70e6bee9cb2ll, 0xee25466956363179ll, 0xfd11ac5d51563753ll, 0x58b1fc3af2489a29ll, 0xa82aa19eaac8748ll, 0x51f02a42fec69c5fll, 0x725f87d6edcef39ll, 0xb51e0df1a07df532ll, 0x8e08134f4074abd4ll, 0xbfdfdba93252fa31ll, 0x37c5c7692f917865ll, 0xe0ed9cb3d636f6b3ll, 0x5b83a93f64405366ll, 0xf06a4f823a1989d7ll, 0xb9ff31ce194773bfll, 0x1d98a6d47f5270c2ll, 0xb43486e2eb7e136bll, 0x64d788eb8413f93dll, 0x9bb30eb73e93d3b2ll, 0xf309ee64e7d4f92ll, 0xb8dde885b32312all, 0x21118ded95ac6f29ll, 0xc983dab25faec20bll, 0x7b74ac6b8d452c1bll, 0xcaccbe26c46e1ea2ll, 0xfeab89e32a652279ll, 0x81b537578dece23dll, 0x714ece790da7d5ebll, 0x771bbd884024ee48ll, 0x6881642b2437ea69ll, 0x72a353465dbfc234ll, 0x2e7739c8b3284985ll, 0x518b6ce43f74b9fdll, 0x1d11b310f142340cll, 0xb8b5f8a7047cc78bll, 0xc98112a0eccee6aell, 0xd4a12b5796e5b7a0ll, 0x24d6431fd4b6ea05ll, 0xafa380a9e9478faall, 0x98a35a0f382f2211ll, 0x608d3b1db228e853ll, 0xec7ada4b8c2e0277ll, 0x59b28e0ee1d21e1cll, 0xc0a93100d57028call, 0xb2ee8e366d7a78d3ll, 0x409e61311cac105bll, 0x152478a51a53cec0ll, 0xa6a22e64bcf36c49ll, 0x32130c6462739094ll, 0x915ff7ea83bb5e2ll, 0xcc03a67084696f4cll, 0xe208ec68ca092003ll, 0xdb6bf75f2140aa60ll, 0xef36c6cc11586b3cll, 0x8afb419de213ac3dll, 0x5e382f8e520c04e1ll, 0x5f92c6bc34ebbb89ll, 0x398db743bfb374fall, 0xe19e0e6e3ceeaa42ll, 0x9e7ea5242816d0d6ll, 0x11dcaa44038fce38ll, 0xc6e3725ac42e00a4ll, 0x71443ce92e90d37cll, 0x23dd572671f8b692ll, 0xde27f7bde1a28bb9ll, 0xe3fc1cc34a75f2ddll, 0x8e5380a90833dfd6ll, 0x4da2b42de56731fdll, 0x36a6009f9eee69bcll, 0xf4db711db3e119ccll, 0x5e8ccc990795a889ll, 0xafd703419dfea9a0ll, 0x30463f3c6ee2b046ll, 0x9e2b31784a512bbll, 0xe94ad634e74bcf82ll, 0x45f3db452cd7f9bbll, 0xde6b6252dd33782bll, 0x193d5f6c04a27310ll, 0x3acb8fa99c329707ll, 0x22d69badb37471f0ll, 0xf1c99f10c56b7cf5ll, 0xfe6110920f406657ll, 0x2258d6aa622458d9ll, 0xd635d70116f7d3fcll, 0x623d3502b380a0fell, 0x967638214e7752d8ll, 0x8ebe4d0ebc1b3a36ll, 0x4693a1c02ec948b3ll, 0xd30995eb3700fd54ll, 0x703135b40eee5981ll, 0x47513b99adac2c47ll, 0x4807b69ef64a62e3ll, 0x351d3b0f620edf57ll, 0xe2d20985d347dddfll, 0xcf1d0224f5f5ba61ll};

unsigned int dx2[] = {0x90a25e6d, 0x266b1fdd, 0xc7e5f9b1, 0x8b331f76, 0x86c26260, 0x11602198, 0x3d449367, 0xe221abb8, 0x20512bec, 0xee5d4140, 0x7a4a63ea, 0x9c9a8137, 0x4518e71d, 0x90bb01b9, 0xc8a0dbb1, 0x1c5859f8, 0xb71d74a2, 0xe627b83d, 0x63ad710f, 0x1b0590aa, 0x47f31b7d, 0x8b247df8, 0x680ceb24, 0x84c82aa9, 0xe16a482, 0x1b303012, 0xa6208757, 0x62842d6f, 0x27ae4f0c, 0x893d6f09, 0xa4687085, 0xa1ffc667, 0xea82a239, 0x17d73135, 0xcb5a9d69, 0x9fb058fc, 0x2998e75c, 0xf0fc9d2, 0xfb1c8ce, 0x79e3bc7c, 0x231366b5, 0x40392f2, 0x3a096a07, 0xd3b975cf, 0xfc49c863, 0xd9b437c6, 0xfea912dc, 0x67224518, 0x92ea1d92, 0x32594898, 0x7436f88d, 0x9a3b2801, 0x54930fd, 0x9273f995, 0x3b9bdecb, 0x44a0665f, 0xde53d2e8, 0xe611f8a1, 0x47dd26d5, 0x62006088, 0x18adef96, 0x40a57445, 0x96d64655, 0xc1c91992, 0xd8d12730, 0x7aeda909, 0x85eaf4c4, 0x60f20c05, 0xc1ed3293, 0xc322ca3f, 0x4c7ed042, 0x73c2b31d, 0xe0396b9d, 0x3020d47f, 0xd4c5d6c6, 0x523f71c0, 0x8ce47a37, 0x7859e90a, 0x7686e532, 0xdb086e00, 0x909ac2c3, 0x574ed015, 0x9e6fa778, 0x54ecf2a9, 0x2db98885, 0xec671188, 0xc0de2005, 0x5ec92a71, 0x78b628e2, 0x9c9816e2, 0x69e13f3d, 0x4fc3827b, 0xae393a9e, 0x6f45648, 0xa85e2113, 0xa2b591bc, 0xaaca0b99, 0x24033483, 0xf90f13b2, 0xa29a99cc};

unsigned int dy2[] = {0x90a25e6d, 0x266b1fdd, 0xc7e5f9b1, 0x8b331f76, 0x86c26260, 0x11602198, 0x3d449367, 0xe221abb8, 0x20512bec, 0xee5d4140, 0x7a4a63ea, 0x9c9a8137, 0x4518e71d, 0x90bb01b9, 0xc8a0dbb1, 0x1c5859f8, 0xb71d74a2, 0xe627b83d, 0x63ad710f, 0x1b0590aa, 0x47f31b7d, 0x8b247df8, 0x680ceb24, 0x84c82aa9, 0xe16a482, 0x1b303012, 0xa6208757, 0x62842d6f, 0x27ae4f0c, 0x893d6f09, 0xa4687085, 0xa1ffc667, 0xea82a239, 0x17d73135, 0xcb5a9d69, 0x9fb058fc, 0x2998e75c, 0xf0fc9d2, 0xfb1c8ce, 0x79e3bc7c, 0x231366b5, 0x40392f2, 0x3a096a07, 0xd3b975cf, 0xfc49c863, 0xd9b437c6, 0xfea912dc, 0x67224518, 0x92ea1d92, 0x32594898, 0x7436f88d, 0x9a3b2801, 0x54930fd, 0x9273f995, 0x3b9bdecb, 0x44a0665f, 0xde53d2e8, 0xe611f8a1, 0x47dd26d5, 0x62006088, 0x18adef96, 0x40a57445, 0x96d64655, 0xc1c91992, 0xd8d12730, 0x7aeda909, 0x85eaf4c4, 0x60f20c05, 0xc1ed3293, 0xc322ca3f, 0x4c7ed042, 0x73c2b31d, 0xe0396b9d, 0x3020d47f, 0xd4c5d6c6, 0x523f71c0, 0x8ce47a37, 0x7859e90a, 0x7686e532, 0xdb086e00, 0x909ac2c3, 0x574ed015, 0x9e6fa778, 0x54ecf2a9, 0x2db98885, 0xec671188, 0xc0de2005, 0x5ec92a71, 0x78b628e2, 0x9c9816e2, 0x69e13f3d, 0x4fc3827b, 0xae393a9e, 0x6f45648, 0xa85e2113, 0xa2b591bc, 0xaaca0b99, 0x24033483, 0xf90f13b2, 0xa29a99cc};


void make_uint288(uint288* out, uint* in, int len) {
    int blocks = len / 9;
    unsigned int rand = 0;
    for (int i = 0; i < blocks; i++) {
        for (int j = 0; j < 9; j++) {
            for (int k = 0; k < 8; k++, rand++) {
                // ATTENTION!! UINT288 is big-endian, so least significant is data[8].
                out[i * 9 + j].data[k + 1] = in[j * 8 + k] + rand;
            }
            out[i * 9 + j].data[0] = 0;
        }
    }
    for (int start = blocks * 9; start < len; start++) {
        out[start].data[8] = 0x3f3fffff + (rand++); // for intergrity.
    }
    //int start = blocks * 
    printf("check uint288: ");
    for (int i = 0; i < 9; i++) {
        printf("0x%llx ", out[0].data[i]);
    }
    printf("\n");
}



#ifndef dh_mybig_copy
#define dh_mybig_copy(a,b) {(a)[0]=(b)[0];(a)[1]=(b)[1];(a)[2]=(b)[2];(a)[3]=(b)[3];}
#endif

__host__ __device__  void my_point_copy(Jpoint* from, Jpoint* to) {
    dh_mybig_copy(to->x, from->x);
    dh_mybig_copy(to->y, from->y);
    dh_mybig_copy(to->x, from->z);
}

void print_big_arr(UINT64* nums,int n){
    for(int i=0;i<n;i++){
        h_mybig_print(nums+i*4);
        printf("\n");
    }
}
void print_jpoint_arr(Jpoint* nums,int n){
    for(int i=0;i<n;i++){
        h_print_pointJ(nums+i);
        printf("\n");
    }
}

void init_Jpoint_toOne(Jpoint* p, int n){
    for(int i=0;i<n;i++){
        for(int j=0;j<4;j++){
            p[i].x[j] = h_Gx[j];
            p[i].y[j] = h_Gy[j];
            p[i].z[j] = h_ONE[j];
        }
    }
}

void make_data(Jpoint *in, int datasize) {
    int block = 31;
    int blocknum = datasize / 31;
    int other = datasize - blocknum * 31;
    printf("datasize = %d\n", datasize);
    unsigned int rand = 1;
    for (int i = 0; i < blocknum; i++) {
        for (int j = 0; j < block; j++, rand++) {
            int idx = i * block + j;
            
            int wtf = (j / 2) * 4; // att: can't simplify this.
            if (j % 2 == 1) {
                dh_mybig_copy(in[idx].x, px+j);
                dh_mybig_copy(in[idx].y, py+j);
            } else {
                dh_mybig_copy(in[idx].x, px3+j);
                dh_mybig_copy(in[idx].y, py3+j);
            }
            //in[idx].y[0] += rand;
        }
    }
    for (int start = blocknum * 31; start < datasize; start++) {
        dh_mybig_copy(in[start].x, px);
        dh_mybig_copy(in[start].y, py);
    }
    printf("makes data\n");
}

void __global__ test_point_double(Jpoint *p1,Jpoint *p2){//only for warm-up.
    UINT64 zinv[4], zinv2[4];
    int tx = threadIdx.x;
    ppoint_double(p1+tx, p2+tx);
    dh_mybig_moninv((p2+tx)->z, zinv);
    dh_mybig_monmult_64(zinv, dc_R2, zinv);
    dh_mybig_monmult_64(zinv, zinv, zinv2);
    dh_mybig_monmult_64((p2+tx)->x, zinv2, (p2+tx)->x);
    dh_mybig_monmult_64(zinv, zinv2, zinv);
    dh_mybig_monmult_64((p2+tx)->y, zinv, (p2+tx)->y);
}

// these functions are not jp1 -> jp2(mont), but jp1 -> jp1(mont), jp2 -> jp2(mont).
void __global__ point_to_monjj(Jpoint* jp1,Jpoint* jp2){
    // int tx = threadIdx.x;
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if(idx<N_BIGNUM){
        dh_mybig_monmult_64((jp1+idx)->x,dc_R2,(jp1+idx)->x);
        dh_mybig_monmult_64((jp1+idx)->y,dc_R2,(jp1+idx)->y);
        dh_mybig_monmult_64((jp1+idx)->z,dc_R2,(jp1+idx)->z);

        dh_mybig_monmult_64((jp2+idx)->x,dc_R2,(jp2+idx)->x);
        dh_mybig_monmult_64((jp2+idx)->y,dc_R2,(jp2+idx)->y);
        dh_mybig_monmult_64((jp2+idx)->z,dc_R2,(jp2+idx)->z);
    }

}

void __global__ point_from_monjj(Jpoint* jp1,Jpoint* jp2){
    // int tx = threadIdx.x;
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if(idx<N_BIGNUM){
        dh_mybig_monmult_64((jp1+idx)->x,dc_ONE,(jp1+idx)->x);
        dh_mybig_monmult_64((jp1+idx)->y,dc_ONE,(jp1+idx)->y);
        dh_mybig_monmult_64((jp1+idx)->z,dc_ONE,(jp1+idx)->z);

        dh_mybig_monmult_64((jp2+idx)->x,dc_ONE,(jp2+idx)->x);
        dh_mybig_monmult_64((jp2+idx)->y,dc_ONE,(jp2+idx)->y);
        dh_mybig_monmult_64((jp2+idx)->z,dc_ONE,(jp2+idx)->z);
    }

}

__global__ void point_from_mont(Jpoint* from, Jpoint* to) {
    UINT64 zinv[4], zinv2[4];
    int tx = threadIdx.x;
    // jacobian->affine
    dh_mybig_moninv((from+tx)->z, zinv);
    dh_mybig_monmult_64(zinv, dc_R2, zinv);
    dh_mybig_monmult_64(zinv, zinv, zinv2);
    dh_mybig_monmult_64((from+tx)->x, zinv2, (to+tx)->x);
    dh_mybig_monmult_64(zinv, zinv2, zinv);
    dh_mybig_monmult_64((from+tx)->y, zinv, (to+tx)->y);
    // mont->normal
    dh_mybig_monmult_64((to+tx)->x,dc_ONE,(to+tx)->x);
    dh_mybig_monmult_64((to+tx)->y,dc_ONE,(to+tx)->y);
    dh_mybig_monmult_64((to+tx)->z,dc_ONE,(to+tx)->z);
}

__global__ void accumulate_sum_per_block(Jpoint* in) { //in = out[]
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int nthread = blockDim.x;
    if (bx == 512) {
        dh_ellipticAdd_JJ(&in[0], &in[N_BIGNUM-1], &in[0]);
        return;
    } 
    in = in + bx * nthread * 2;
     for (int i = nthread, j = 1; i; i /= 2, j++) { // 5 = log2(thread)
        #ifdef DEBUG
                if (tx == 0) {
                    //printf("tx=%d, point=%llx\n", tx, in[tx].x[0]);
                }
        #endif
                if (tx < i) {
                    
                    // printf("tx=%d, before add: point1=%llx, point2=%llx\n", tx, in[tx].x[0], in[tx + i].x[0]);
        #ifdef VERBOSE_MODE
                    dh_ellipticAdd_JJ_verbose(&in[tx], &in[tx + i], &in[tx]);
        #else
                    dh_ellipticAdd_JJ(&in[tx], &in[tx + i], &in[tx]);
        #endif
                    // printf("tx=%d, after add: point=%llx\n", tx, in[tx].x[0]);
                }
                __syncthreads();
        }
    
   
       
}

__global__ void accumulate__blocks(Jpoint* in, int blocksize) { //in = out[]
    //int bx = blockIdx.x;
    int tx = threadIdx.x; // tx is last step's block id
    int nthread = blockDim.x;
    // in = in + tx * blocksize;
    for (int i = nthread, j = 1; i; i /= 2, j++) { // 5 = log2(thread)
#ifdef DEBUG
        if (tx == 0) {
            //printf("tx=%d, point=%llx\n", tx, in[tx].x[0]);
        }
#endif
        if (tx < i) {
            // printf("tx=%d, before add: point1=%llx, point2=%llx\n", tx, (in + tx * blocksize)->x[0], (in + (tx + i) * blocksize)->x[0]);
#ifdef VERBOSE_MODE
            dh_ellipticAdd_JJ_verbose(&in[tx], &in[tx + i], &in[tx]);
#else
            dh_ellipticAdd_JJ(in + tx * blocksize , in + (tx + i) * blocksize, in + tx * blocksize);
#endif
            // printf("tx=%d, after add: point=%llx\n", tx, (in + tx * blocksize)->x[0]);
        }
        __syncthreads();
    }
}

__global__ void trivial_sum(Jpoint* in, int num) {
    for (int i = 1; i < num; i++) {
        // printf("do algorithm: i=%d\n", i);
        dh_ellipticAdd_JJ(in, in + i, in);
        //__syncthreads__();
    }
}

__global__ void trivial_sum_blocks(Jpoint* in, int num, int blocksize) {
    for (int i = 1; i < num; i++) {
        // printf("do algorithm: i=%d\n", i);
        dh_ellipticAdd_JJ(in, in + i * blocksize, in);
        //__syncthreads__();
    }
}


#define from(x, y, z) *(from + (x) * DBC_level1 + (y) * DBC_level2 + (z))
const int DBC_level1 = 3 * DBC_MAXLENGTH;
const int DBC_level2 = 3;
__global__ void check_dbc(uint288* nums, int* dbc_store, int* dbc_value) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int nthread = blockDim.x;
    int dbc_id = nthread * bx + tx;

    int* from = dbc_store + dbc_id * 6 * DBC_MAXLENGTH;
    if (tx == 120) {
        printf("check viability: ");
        for (int i = 0; i < 120; i++) {
            printf("0x%x ", u_pow23[0][i].data[8]);
        }
        printf("\n");
        //printf("step0: checkdata, tx=%d, bx=%d, nth=%d, data=0x%x\n", tx, bx, nthread, (unsigned int)u_pow23);
        //printf("step0: check_dbc, tx=%d, bx=%d, nthread=%d\n", threadIdx, blockIdx, nthread);
    }
    if (nums->data[8] == 1) {
        return;
    }
    if(nums->data[0] != 0) {
        printf("panic: thread %d has illegal input\n", tx);
    }
    __syncthreads();
    int n = get_DBC(nums + dbc_id, from, dbc_value + dbc_id * 2);
    // printf("thread %d returns %d\n", tx, n);
    // return;
    //__syncthreads();
    //get_DBC(nums + dbc_id, from, dbc_value + dbc_id * 2);
    for (int i = 0; i < DBC_MAXLENGTH; i++) {
        for (int j = 0; j < 3; j++) {
            from(0, i, j) = from(n, i, j);
        }
    }
    dbc_value[dbc_id * 2] = dbc_value[dbc_id * 2 + n];
}

__device__ void my_check_point(Jpoint *pt) {
    printf("[0x%x, 0x%x, 0x%x]\n", pt->x[0], pt->y[0], pt->z[0]);
}
#define RELEASE
__global__ void dbc_main(uint288* nums, int* dbc_store, int* dbc_value, Jpoint *in, Jpoint *out) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int nthread = blockDim.x;
    int dbc_id = nthread * bx + tx;
    int n = 0;
    if (dbc_id == 0) {
        // do nothing
    }
    else if (bx < CONST_BLOCK - 1 && dbc_id < N_BIGNUM) {
        n = get_DBC(nums + dbc_id, dbc_store + dbc_id * 6 * DBC_MAXLENGTH, dbc_value + dbc_id * 2);
    }
    int len = *(dbc_value + dbc_id * 2 + n);
    int* dbc = dbc_store + dbc_id * 6 * DBC_MAXLENGTH + n * 3 * DBC_MAXLENGTH;
#ifndef RELEASE
     __syncthreads();
    if(nums->data[0] != 0 || len < 1 || len > 110) {
        printf("panic: block %d thread %d has illegal input\n", bx, tx);
    }
    if (tx == 1) {
        printf("check dbc value in block %d, len = %d: \n", bx, len);
        for (int j = 0; j < len; j++) {
            int* wtf = dbc + j * 3;
            printf("(%d)*(2**%d)*(3**%d) + ", wtf[0], wtf[1], wtf[2]);
        }
        printf("\n");
    }
    __syncthreads();
#endif
    int cnt = 0;
    if (dbc_id == 0) {
        // 
        dh_my_point_copy(in + dbc_id, out + dbc_id);
    } else if ((bx < CONST_BLOCK - 1) && dbc_id < N_BIGNUM) {
         cnt = run_DBC_v2(in + dbc_id, out + dbc_id, dbc, len);
    }
    // if (tx == 0) {
    //     printf("bx=%d runs %d ops, check point value\n", bx, cnt);
    //     printf("{[%d]2^%d 3^%d} ", dbc[0], dbc[1], dbc[2]);
    //     my_check_point(in+dbc_id);
    //     //my_check_point(out+dbc_id);
    // }
    //__syncthreads();
    //printf("(%d %d) ", bx, tx);
}

unsigned int test_scalars[][8] = {
{0xc8ba6c0b, 0xf4e7df38, 0xdbda17e, 0x82c20ef0, 0x1055f4fd, 0x76121c9, 0x40b8cf2c, 0xd0c3935, },
{0xa7e9ce6f, 0x21b8a7b7, 0x3cc041d2, 0x28793782, 0xe78268bc, 0xa4788ca0, 0xbf991122, 0x5cd20a2, },
{0xfb329170, 0xaa7d86ca, 0xa6fc64dd, 0x3629f4c1, 0xb3ff14bb, 0x2e595ec0, 0xa8645a9a, 0x2f9d3c73, },
{0x6a40bc64, 0x985b8258, 0x4f2f58de, 0x80994d7b, 0x84af59c5, 0x91ed914f, 0xc13c3c03, 0x130e3fa1, },
{0xe895f51b, 0xcab448ac, 0xb31f4f78, 0xd3e6ecc3, 0x5aa6266e, 0x93931e03, 0xcf8c65f6, 0x19acacc6, },
{0x912deadb, 0xa1fd83a4, 0x913a6692, 0xc55c504b, 0x3ca1f176, 0x3ae5f26d, 0x982ef124, 0x2067044, },
{0x3f775df8, 0x59b3e897, 0x847d2ba8, 0x83b012d5, 0xf4a7fb0f, 0x81697b01, 0xcd24dd12, 0x270d1ed5, },
{0xa88c61be, 0xb0569c3c, 0x8ce7b4cf, 0xf48393d2, 0x2d8d52a2, 0x81a0496, 0x3d1b1bbf, 0x2705079a, },
};

UINT64 test_P_x[][4] = {
    {0x67277bfb1bf5a2b3ULL, 0x748a78f2c0409600ULL, 0x5591098a24842f91ULL, 0x12626df20be795acULL},
    {0x96b8a79d4e7b2fa0ULL, 0xa7b255b7ce68b7d1ULL, 0x748d51201c6bc97ULL, 0xedd668431802799ULL},
    {0xdef8a6ee1b1ee6fdULL, 0x421897ed0b53689cULL, 0x4eab68e7d44f2f90ULL, 0x1e2b172af72f5598ULL},
    {0x6ba406241ec03abfULL, 0x4ac6af3893510bb3ULL, 0xf2ac4eb7d195380bULL, 0x6de870c977240cULL},
    {0x9d453953d29b791fULL, 0x3e21372b6d330e94ULL, 0x33f52c13f38afa1fULL, 0x29b073c75fb72771ULL},
    {0x5168b89708f5e1ebULL, 0xd2668b78e993750dULL, 0x2edace5f7678bcf1ULL, 0x5b8fa45091ab9e1ULL},
    {0x25d86e3d59542496ULL, 0xe2a8e448860f425dULL, 0x7d259942ea413630ULL, 0x1be50b1000752ffdULL},
    {0x10b30dbe33895a85ULL, 0xce23b30959e50cfaULL, 0xb9844001735f9518ULL, 0x271e9140aa677758ULL},
};

UINT64 test_P_y[][4] = {
    {0x150620a3cef1c399ULL, 0x8c96ca0019f2e7e5ULL, 0x10f275339bbfae51ULL, 0x1bad4003b802c9f0ULL},
    {0x7ac3a975b78f95a1ULL, 0xadfe813cec94508eULL, 0x6200f7a5f012802fULL, 0x22105cbf9fa748b6ULL},
    {0x14f575b382668579ULL, 0x2bd5a70a46be0b28ULL, 0x3eefe424a72226d6ULL, 0xb193a5e58e61fc5ULL},
    {0x4deddbd4c1e9a593ULL, 0x1128bf643f129ca8ULL, 0xcdab7431aa096ad7ULL, 0x2e52ec348f885c0cULL},
    {0xd6a36c9b5f868b69ULL, 0x9558c02e3a5baeaeULL, 0xb492ae1a12637ff7ULL, 0x217ca8ee989a5d43ULL},
    {0x84cf600640672428ULL, 0x40ca230e51fcc345ULL, 0x523a2d9de4c48d1aULL, 0x2acf4d5231cb7411ULL},
    {0xb2458c27d6900e1cULL, 0xc06d987d52dc3320ULL, 0xcadb27d8f29fd6c9ULL, 0x22779b1d38ce6d7ULL},
    {0xb4e8c2f9c0db3ea3ULL, 0x72b99a23081ae9a4ULL, 0x6688814cde0fcb84ULL, 0x29550fd01950fa3bULL},
};
UINT64 test_res_P_x[][4] = { 0x11D9DA0F08AFAC7FULL, 0x1DA393D1310C21D4ULL, 0x8CCD46EFD0A1CE33ULL, 0x71ECDE620B917BE9ULL};
UINT64 test_res_P_y[][4] = { 0x2AEDFE94AE0345C6ULL, 0x9D8A9E1F4669D88DULL, 0x146AA8AFE73B769AULL, 0x3D966433CC2FFCF8ULL};

__global__ void test_scalar_multiplication(UINT64* scalar, Jpoint* in, Jpoint* out) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int nthread = blockDim.x;
    int idx = nthread * bx + tx;
    if (tx == 0) {
        printf("Check input data: scalar=%llx, in=%llx\n", scalar[4 * tx], in->x[0]);
    }
    scalar_multiplication(scalar + 4 * idx, in + idx, out + idx);
}


int correctness_test() {
    struct timeval s1,e1;
    long long time_use=1;
    uint288* scalar;
    uint288* d_scalar;
    int* dbc_store_host;
    int* dbc_len_host;
    int* dbc_store_device;
    int* dbc_len_device;

    Jpoint* w_p1;
    Jpoint* w_p2;
    Jpoint* wd_p1;
    Jpoint* wd_p2;
    Jpoint* d_p1;
    Jpoint* d_p2;
    Jpoint* t_p1;
    Jpoint* t_p2;
    Jpoint* td_p1;
    Jpoint* td_p2;
    UINT64* trivial_scalar;
    UINT64* h_trivial_scalar;

    
    
    CUDA_SAFE_CALL(cudaMalloc((void**)&dbc_store_device, 6 * DBC_MAXLENGTH * 16 * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&dbc_len_device, 2 * 16 * sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(dbc_store_device, 0, sizeof(dbc_store_device)));
    
    //CUDA_SAFE_CALL(cudaMemcpy(d_dbc, dbc, sizeof(DBC)*N_BIGNUM, cudaMemcpyHostToDevice));
    
    printf("data required\n");
    // point init
    w_p1 = (Jpoint*)malloc(100 * sizeof(Jpoint));
    w_p2 = (Jpoint*)malloc(100 * sizeof(Jpoint));
    for (int i = 0; i < 100; i++) {
        for(int j = 0;j < 4; j++){
            w_p1[i].x[j] = px[i*4+j];
            w_p1[i].y[j] = py[i*4+j];
            w_p1[i].z[j] = h_ONE[j];
        }
    }

    t_p1 = (Jpoint*)malloc(16 * sizeof(Jpoint));
    t_p2 = (Jpoint*)malloc(16 * sizeof(Jpoint));
    for (int i = 0; i < 8; i++) {
        for(int j = 0; j < 4; j++){
            t_p1[i].x[j] = test_P_x[i][j];
            t_p1[i].y[j] = test_P_y[i][j];
            t_p1[i].z[j] = h_ONE[j];
        }
    }
    scalar = (uint288*)malloc(16*sizeof(uint288));
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            scalar[i].data[8 - j] = test_scalars[i][j];
        }
        scalar[i].data[0] = 0;
    }
    trivial_scalar = (UINT64*)malloc(16 * 4 * sizeof(UINT64));
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 4; j++) {
            if (j == 0) trivial_scalar[4 * i] = 3;
            else trivial_scalar[4 * i + j] = 0; 
            trivial_scalar[4 * i + j] = test_P_x[0][j];
        }
    }
    printf("data inited\n");
    
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_p1,16*sizeof(Jpoint)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_p2,16*sizeof(Jpoint)));
    CUDA_SAFE_CALL(cudaMemcpy(d_p1,t_p1,16*sizeof(Jpoint),cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_p2,t_p2,16*sizeof(Jpoint),cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_scalar,sizeof(uint288)*16));
    CUDA_SAFE_CALL(cudaMemcpy(d_scalar, scalar, 16*sizeof(uint288),cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMalloc((void**)&h_trivial_scalar, 16 * 4 * sizeof(UINT64)));
    CUDA_SAFE_CALL(cudaMemcpy(h_trivial_scalar, trivial_scalar, 16 * 4 * sizeof(UINT64), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMalloc((void**)&wd_p1,100*sizeof(Jpoint)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&wd_p2,100*sizeof(Jpoint)));
    CUDA_SAFE_CALL(cudaMemcpy(wd_p1,w_p1,100*sizeof(Jpoint),cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(wd_p2,w_p2,100*sizeof(Jpoint),cudaMemcpyHostToDevice));
    printf("ready\n");

    // warm-up.
    // attention: even small data, test data needs warm-up. Else the output would be random.
    test_point_double<<<1, 100>>>(wd_p1, wd_p2);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR();
    printf("warm-up ends.\n");


    // go test.
    // step 1: point to montgomery(jacobian is default format).
    point_to_monjj<<<1, 8>>>(d_p1,d_p2);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR();
    
#ifdef DEBUG
    printf("dbc dp1:\n");
    CUDA_SAFE_CALL(cudaMemcpy(t_p1,d_p1,16*sizeof(Jpoint),cudaMemcpyDeviceToHost));
    for (int i = 0; i < 8; i++) {
        printf("0x%llx 0x%llx 0x%llx 0x%llx\n", t_p1[i].x[3], t_p1[i].x[2], t_p1[i].x[1], t_p1[i].x[0]);
    }
#endif
    //test_point_double<<<1, 8>>>(d_p1, d_p2);
    dbc_main<<<1,8>>>(d_scalar, dbc_store_device, dbc_len_device, d_p1, d_p2);
    // test_scalar_multiplication<<<1, 8>>>(h_trivial_scalar, d_p1, d_p2);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR();

#ifdef DEBUG
    printf("dbc ans on mont:\n");
    CUDA_SAFE_CALL(cudaMemcpy(t_p2,d_p2,16*sizeof(Jpoint),cudaMemcpyDeviceToHost));
    for (int i = 0; i < 8; i++) {
        printf("0x%llx 0x%llx 0x%llx 0x%llx\n", t_p2[i].x[3], t_p2[i].x[2], t_p2[i].x[1], t_p2[i].x[0]);
    }
#endif
    // just check dbc_main
    // point_from_monjj<<<1, 8>>>(d_p1, d_p2);
    point_from_mont<<<1, 8>>>(d_p2, d_p1);
    // cudaDeviceSynchronize();
    // CUDA_CHECK_ERROR();

#ifdef DEBUG
    printf("dbc single ans's:\n");
    CUDA_SAFE_CALL(cudaMemcpy(t_p2,d_p1,16*sizeof(Jpoint),cudaMemcpyDeviceToHost));
    for (int i = 0; i < 8; i++) {
        printf("0x%llx 0x%llx 0x%llx 0x%llx\n", t_p2[i].x[3], t_p2[i].x[2], t_p2[i].x[1], t_p2[i].x[0]);
    }
#endif

    // accumulate_sum<<<1, 4>>>(d_p2);
    // trivial_sum<<<1, 1>>>(d_p2, 8);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR();

    CUDA_SAFE_CALL(cudaMemcpy(t_p1,d_p2,16*sizeof(Jpoint),cudaMemcpyDeviceToHost));
    printf("point x: 0x%llx 0x%llx 0x%llx 0x%llx\n", t_p1[0].x[3], t_p1[0].x[2], t_p1[0].x[1], t_p1[0].x[0]);
    printf("point y: 0x%llx 0x%llx 0x%llx 0x%llx\n", t_p1[0].y[3], t_p1[0].y[2], t_p1[0].y[1], t_p1[0].y[0]);
    printf("point z: 0x%llx 0x%llx 0x%llx 0x%llx\n", t_p1[0].z[0], t_p1[0].z[1], t_p1[0].z[2], t_p1[0].z[3]);

    point_from_mont<<<1, 8>>>(d_p2, d_p1);
    //point_from_monjj<<<1, 8>>>(d_p1,d_p2);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR();

    printf("dbc sum ans:\n");
    CUDA_SAFE_CALL(cudaMemcpy(t_p1,d_p1,16*sizeof(Jpoint),cudaMemcpyDeviceToHost));
    printf("0x%llx 0x%llx 0x%llx 0x%llx\n", t_p1[0].x[3], t_p1[0].x[2], t_p1[0].x[1], t_p1[0].x[0]);
    printf("0x%llx 0x%llx 0x%llx 0x%llx\n", t_p1[0].z[0], t_p1[0].z[1], t_p1[0].z[2], t_p1[0].z[3]);
}





// 数据准备
template<typename ppT>
class groth16_parameters {
  public:
    size_t d;
    size_t m;
    std::vector<G1<ppT>> A, B1, L, H;
    std::vector<G2<ppT>> B2;

  groth16_parameters(const char* path) {
    FILE* params = fopen(path, "r");
    d = read_size_t(params);
    m = read_size_t(params);
    for (size_t i = 0; i <= m; ++i) { A.emplace_back(read_bn128_g1<ppT>(params)); }
    for (size_t i = 0; i <= m; ++i) { B1.emplace_back(read_bn128_g1<ppT>(params)); }
    for (size_t i = 0; i <= m; ++i) { B2.emplace_back(read_bn128_g2<ppT>(params)); }
    for (size_t i = 0; i < m-1; ++i) { L.emplace_back(read_bn128_g1<ppT>(params)); }
    for (size_t i = 0; i < d; ++i) { H.emplace_back(read_bn128_g1<ppT>(params)); }
    fclose(params);
  }
};

template<typename ppT>
class groth16_input {
  public:
    std::vector<Fr<ppT>> w;
    std::vector<Fr<ppT>> ca, cb, cc;
    Fr<ppT> r;

  groth16_input(const char* path, size_t d, size_t m) {
    FILE* inputs = fopen(path, "r");

    for (size_t i = 0; i < m + 1; ++i) { w.emplace_back(read_fr<ppT>(inputs)); }

    for (size_t i = 0; i < d + 1; ++i) { ca.emplace_back(read_fr<ppT>(inputs)); }
    for (size_t i = 0; i < d + 1; ++i) { cb.emplace_back(read_fr<ppT>(inputs)); }
    for (size_t i = 0; i < d + 1; ++i) { cc.emplace_back(read_fr<ppT>(inputs)); }

    r = read_fr<ppT>(inputs);

    fclose(inputs);
  }
};

template<typename ppT>
class groth16_output {
  public:
    G1<ppT> A, C;
    G2<ppT> B;

  groth16_output(G1<ppT> &&A, G2<ppT> &&B, G1<ppT> &&C) :
    A(std::move(A)), B(std::move(B)), C(std::move(C)) {}

  void write(const char* path) {
    FILE* out = fopen(path, "w");
    write_bn128_g1<ppT>(out, A);
    write_bn128_g2<ppT>(out, B);
    write_bn128_g1<ppT>(out, C);
    fclose(out);
  }
};



//计算多项式H(x)
template<typename ppT>
std::vector<Fr<ppT>> compute_H(
    size_t d,
    std::vector<Fr<ppT>> &ca,
    std::vector<Fr<ppT>> &cb,
    std::vector<Fr<ppT>> &cc)
{
    // Begin witness map
    libff::enter_block("Compute the polynomial H");

    const std::shared_ptr<libfqfft::evaluation_domain<Fr<ppT>> > domain = libfqfft::get_evaluation_domain<Fr<ppT>>(d + 1);

    domain->iFFT(ca);
    domain->iFFT(cb);

    domain->cosetFFT(ca, Fr<ppT>::multiplicative_generator);
    domain->cosetFFT(cb, Fr<ppT>::multiplicative_generator);

    libff::enter_block("Compute evaluation of polynomial H on set T");
    std::vector<Fr<ppT>> &H_tmp = ca; // can overwrite ca because it is not used later
#ifdef MULTICORE
#pragma omp parallel for
#endif
    for (size_t i = 0; i < domain->m; ++i)
    {
        H_tmp[i] = ca[i]*cb[i];
    }
    std::vector<Fr<ppT>>().swap(cb); // destroy cb

    domain->iFFT(cc);

    domain->cosetFFT(cc, Fr<ppT>::multiplicative_generator);

#ifdef MULTICORE
#pragma omp parallel for
#endif
    for (size_t i = 0; i < domain->m; ++i)
    {
        H_tmp[i] = (H_tmp[i]-cc[i]);
    }

    domain->divide_by_Z_on_coset(H_tmp);

    libff::leave_block("Compute evaluation of polynomial H on set T");

    domain->icosetFFT(H_tmp, Fr<ppT>::multiplicative_generator);

    std::vector<Fr<ppT>> coefficients_for_H(domain->m+1, Fr<ppT>::zero());
#ifdef MULTICORE
#pragma omp parallel for
#endif
    for (size_t i = 0; i < domain->m; ++i)
    {
        coefficients_for_H[i] = H_tmp[i];
    }

    libff::leave_block("Compute the polynomial H");

    return coefficients_for_H;
}


const multi_exp_method method = multi_exp_method_BDLO12;
// CPU上的点乘计算
template<typename G, typename Fr>
G multiexp(typename std::vector<Fr>::const_iterator scalar_start,
           typename std::vector<G>::const_iterator g_start,
           size_t length)
{
#ifdef MULTICORE
    const size_t chunks = omp_get_max_threads(); // to override, set OMP_NUM_THREADS env var or call omp_set_num_threads()
#else
    const size_t chunks = 1;
#endif

    return libff::multi_exp_with_mixed_addition<G,
                                                Fr,
                                                method>(
        g_start,
        g_start + length,
        scalar_start,
        scalar_start + length,
        chunks);

}




template<typename ppT>
int run_prover(
    const char* params_path,
    const char* input_path,
    const char* output_path) {
    
    cudaSetDevice(0);

#ifdef CORRECTNESS_TEST
    correctness_test();
    return 0;
#endif

    // 初始化曲线参数和公共输入等参数
    ppT::init_public_params();
    printf("曲线初始化完毕\n");
    const size_t primary_input_size = 1;
    const groth16_parameters<ppT> parameters(params_path);
    printf("groth16_parameters参数读取完毕\n");
    const groth16_input<ppT> input(input_path, parameters.d, parameters.m);

    std::vector<Fr<ppT>> w  = std::move(input.w);
    std::vector<Fr<ppT>> ca = std::move(input.ca);
    std::vector<Fr<ppT>> cb = std::move(input.cb);
    std::vector<Fr<ppT>> cc = std::move(input.cc);
    printf("w,ca,cb,cc参数读取完毕\n");
    // 使用FFT/iFFT计算H多项式系数
    std::vector<Fr<ppT>> coefficients_for_H = compute_H<ppT>(
        parameters.d,
        ca, cb, cc);
    printf("多项式H计算完毕\n");
    // 计算椭圆曲线点乘

    // 处理Fr<ppT>的数据，将其转换为uint288的格式

    struct timeval s1,e1;
    long long time_use=1;
    int nB,nT;
    
    cudaOccupancyMaxPotentialBlockSize(&nB,&nT,dbc_main);
    printf("NB=%d,NT=%d\n",nB,nT);
    
    uint288* scalar;
    uint288* d_scalar;
    uint288* scalar_W;
    uint288* scalar_A;
    uint288* scalar_C;
    int* dbc_store_host;
    int* dbc_len_host;
    int* dbc_store_device;
    int* dbc_len_device;

    Jpoint* h_p1;
    Jpoint* h_G1_A;
    Jpoint* h_G1_B1;
    Jpoint* h_G1_C;
    Jpoint* h_p2;
    Jpoint* d_p1;
    Jpoint* d_p2;
    Jpoint* t_p1;
    Jpoint* t_p2;
    Jpoint* td_p1;
    Jpoint* td_p2;

    // number init.
    scalar = (uint288*)malloc((parameters.m+1)*sizeof(uint288));
    
    dbc_store_host = (int*)malloc(6 * DBC_MAXLENGTH * (parameters.m+1) * sizeof(int)); // dbc_store[2][DBC_MAXSIZE][3];
    dbc_len_host = (int*)malloc(2 * (parameters.m+1) * sizeof(int)); // dbc_len[2];
    int dbc_size = 6 * DBC_MAXLENGTH;
    //生成1024个uint288的数
    // make_uint288(scalar, dx2, (parameters.m+1)); // init int288
    scalar_W = (uint288*)malloc((parameters.m+1)*sizeof(uint288));
    // 处理scalar
    // printf("%llx, %llx,%llx,%llx", w[0].as_bigint().data[0],w[0].as_bigint().data[1],w[0].as_bigint().data[2],w[0].as_bigint().data[3]);
    for (int i = 0; i <= parameters.m; i++) {
        // 将unsigned long表示的64位数转为9位32位表示的uint288的数
        for (int k = 0; k < 8; k=k+2) {
            scalar_W[i].data[k+1] = ((w[i].as_bigint().data[(7-k)/2]>>32)&0xffffffff);
            scalar_W[i].data[k+2] = ((w[i].as_bigint().data[(7-k)/2])&0xffffffff);
        }
        scalar_W[i].data[0] = 0;
    }
    // 处理G1
    h_G1_A = (Jpoint*)malloc((parameters.m+1)*sizeof(Jpoint));
    
    for (int i = 0; i <= parameters.m; i++) {
        auto bn128_Fp_X = parameters.A[i].coord[0];
        auto bn128_Fp_Y = parameters.A[i].coord[1];
        bn::Fp from_x, from_y;
        bn::Fp::fromMont(from_x, bn128_Fp_X);
        bn::Fp::fromMont(from_y, bn128_Fp_Y);
        for (int k = 0; k < 4; k++) {
            h_G1_A[i].x[k] = from_x[k];
            h_G1_A[i].y[k] = from_y[k];
            h_G1_A[i].z[k] = h_ONE[k];
        }
    }

    printf("H_G1_A1:%llx,%llx,%llx,%llx\n", h_G1_A[0].x[0],h_G1_A[0].x[1],h_G1_A[0].x[2],h_G1_A[0].x[3]);
    printf("scalar1:%lx,%lx,%lx,%lx,%lx,%lx,%lx,%lx\n", scalar_W[1].data[1], scalar_W[1].data[2],scalar_W[1].data[3],scalar_W[1].data[4],scalar_W[1].data[5],scalar_W[1].data[6],scalar_W[1].data[7],scalar_W[1].data[8]);
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_scalar,sizeof(uint288)*(parameters.m+1)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&dbc_store_device, 6 * DBC_MAXLENGTH * (parameters.m+1) * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&dbc_len_device, 2 * (parameters.m+1) * sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(dbc_store_device, 0, sizeof(dbc_store_device)));
    //CUDA_SAFE_CALL(cudaMemcpy(d_dbc, dbc, sizeof(DBC)*N_BIGNUM, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_scalar, scalar_W, (parameters.m+1)*sizeof(uint288),cudaMemcpyHostToDevice));

    // point init
    t_p1 = (Jpoint*)malloc(100 * sizeof(Jpoint));
    t_p2 = (Jpoint*)malloc(100 * sizeof(Jpoint));
    // for (int i = 0; i < 100; i++) {
    //     for(int j = 0;j < 4; j++){
    //         t_p1[i].x[j] = px[i*4+j];
    //         t_p1[i].y[j] = py[i*4+j];
    //         t_p1[i].z[j] = h_ONE[j];
    //     }
    // }
    h_p1 = (Jpoint*)malloc((parameters.m+1)*sizeof(Jpoint));
    h_p2 = (Jpoint*)malloc((parameters.m+1)*sizeof(Jpoint));
    // init_Jpoint_toOne(h_p1, (parameters.m+1));
    // make_data(h_p1, (parameters.m+1));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_p1,(parameters.m+1)*sizeof(Jpoint)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_p2,(parameters.m+1)*sizeof(Jpoint)));
    CUDA_SAFE_CALL(cudaMemcpy(d_p1,h_G1_A,(parameters.m+1)*sizeof(Jpoint),cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_p2,h_p2,(parameters.m+1)*sizeof(Jpoint),cudaMemcpyHostToDevice));

    //=== TEST =====
    printf("ready\n");
    gettimeofday(&s1,NULL);
    check_dbc<<<1,128>>>(d_scalar, dbc_store_device, dbc_len_device);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR();

    // CUDA_SAFE_CALL(cudaMemcpy(h_p1,d_p1,N_POINT*sizeof(Jpoint),cudaMemcpyDeviceToHost));
    // CUDA_SAFE_CALL(cudaMemcpy(h_p2,d_p2,N_POINT*sizeof(Jpoint),cudaMemcpyDeviceToHost));
    //print_jpoint_arr(h_p1,1);
    CUDA_SAFE_CALL(cudaMemcpy(dbc_store_host, dbc_store_device, 6 * DBC_MAXLENGTH * (parameters.m+1) * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(dbc_len_host, dbc_len_device,2 * (parameters.m+1) * sizeof(int), cudaMemcpyDeviceToHost));
    gettimeofday(&e1,NULL);
    time_use=(e1.tv_sec-s1.tv_sec)*1000000+(e1.tv_usec-s1.tv_usec);//微秒
    printf("test usage is %ld us\n",time_use);
    
    //==== WARM UP =====
    gettimeofday(&s1,NULL);
    point_to_monjj<<<N_BLOCK,N_THREAD_PER_BLOCK>>>(d_p1,d_p2);

    // testbasemul<<<N_BLOCK,N_THREAD_PER_BLOCK>>>(d_p1,d_num);
    test_point_double<<<N_BLOCK,N_THREAD_PER_BLOCK>>>(d_p1,d_p2);

    point_from_monjj<<<N_BLOCK,N_THREAD_PER_BLOCK>>>(d_p1,d_p2);

    gettimeofday(&e1,NULL);
    time_use=(e1.tv_sec-s1.tv_sec)*1000000+(e1.tv_usec-s1.tv_usec);//微秒
    printf("warm up time usage is %ld us\n",time_use);

    //==== MAIN =====
    gettimeofday(&s1,NULL);

    point_to_monjj<<<N_BLOCK,N_THREAD_PER_BLOCK>>>(d_p1,d_p2);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR();

#ifdef PRECOMPUTE
    multi_scalar_multiple<<<N_BLOCK,N_THREAD_PER_BLOCK>>>(d_dbc, d_p1, d_p2);
#else
    dbc_main<<<N_BLOCK,N_THREAD_PER_BLOCK>>>(d_scalar, dbc_store_device, dbc_len_device, d_p1, d_p2);
#endif

    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR();
    
    point_from_monjj<<<N_BLOCK, N_THREAD_PER_BLOCK>>>(d_p1, d_p2);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR();
    CUDA_SAFE_CALL(cudaMemcpy(t_p1,d_p2,90*sizeof(Jpoint),cudaMemcpyDeviceToHost));

    printf("dbc main res start:\n");
    for (int i = 0; i < 50; i++) {
        printf("index=%d X:0x%llx 0x%llx 0x%llx 0x%llx\n", (i+1), t_p1[i].x[3], t_p1[i].x[2], t_p1[i].x[1], t_p1[i].x[0]);
        printf("index=%d Y:0x%llx 0x%llx 0x%llx 0x%llx\n", (i+1), t_p1[i].y[3], t_p1[i].y[2], t_p1[i].y[1], t_p1[i].y[0]);
    }
    printf("dbc main res end:\n");
    accumulate_sum_per_block<<<N_BLOCK , N_THREAD_PER_BLOCK/2>>>(d_p2);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR();
    
    accumulate__blocks<<<1, N_BLOCK/2>>>(d_p2, N_THREAD_PER_BLOCK);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR();
#ifdef DEBUG
    printf("dbc dp1:\n");
    CUDA_SAFE_CALL(cudaMemcpy(t_p1,d_p1,100*sizeof(Jpoint),cudaMemcpyDeviceToHost));
    for (int i = 0; i < 10; i++) {
        printf("0x%llx 0x%llx 0x%llx 0x%llx\n", t_p1[i].x[0], t_p1[i].x[1], t_p1[i].x[2], t_p1[i].x[3]);
    }
#endif
             
    point_from_mont<<<N_BLOCK,N_THREAD_PER_BLOCK>>>(d_p2,d_p1);
    cudaDeviceSynchronize();

    CUDA_CHECK_ERROR();
    //
    CUDA_SAFE_CALL(cudaMemcpy(t_p1,d_p1,16*sizeof(Jpoint),cudaMemcpyDeviceToHost));
    printf("gpu answer:\n");
    printf("0x%llx 0x%llx 0x%llx 0x%llx\n", t_p1[0].x[3], t_p1[0].x[2], t_p1[0].x[1], t_p1[0].x[0]);
    printf("0x%llx 0x%llx 0x%llx 0x%llx\n", t_p1[0].y[3], t_p1[0].y[2], t_p1[0].y[1], t_p1[0].y[0]);
    printf("0x%llx 0x%llx 0x%llx 0x%llx\n", t_p1[0].z[3], t_p1[0].z[2], t_p1[0].z[1], t_p1[0].z[0]);
    gettimeofday(&e1,NULL);
    time_use=(e1.tv_sec-s1.tv_sec)*1000000+(e1.tv_usec-s1.tv_usec);//微秒
    printf("time_use is %ld us\n",time_use);

    // used in accuracy test.
    // printf("Check DBC ans: \n");
    // for (int i = 0; i < 8; i++) {
    //     printf("\nCheck DBC %d ans: \n", i+1);
    //     for (int j = 0; j < dbc_len_host[2 * i]; j++) {
    //         int* wtf = dbc_store_host + i * 6 * DBC_MAXLENGTH + j * 3;
    //         printf("{[%d]2^%d 3^%d} ", wtf[0], wtf[1], wtf[2]);
    //     }
    // }
    free(scalar);
    free(dbc_store_host);
    //free(h_p1);
    //free(h_p2);
    CUDA_SAFE_CALL(cudaFree(d_scalar));
    CUDA_SAFE_CALL(cudaFree(dbc_store_device));
    CUDA_SAFE_CALL(cudaFree(dbc_len_device));
    CUDA_SAFE_CALL(cudaFree(d_p1));
    CUDA_SAFE_CALL(cudaFree(d_p2));
}



// TODO 将main函数改成一个接受scalar
// #define CORRECTNESS_TEST
int main(int argc, const char* argv[]){
    const char *params_path = argv[1];
    const char *input_path = argv[2];
    const char *output_path = argv[3];
    printf("%s %s %s\n", params_path, input_path, output_path);
    run_prover<bn128_pp>(params_path, input_path, output_path);
    return 0;
}
