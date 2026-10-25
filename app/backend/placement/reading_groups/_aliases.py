"""Questions that cover the same concept as another question share its verified readings.

Each key is a question id; the value is the id whose readings it reuses.
"""
ALIASES = {
    # classical ML "from scratch" rounds reuse the concept readings of the coding questions
    638: 142, 955: 142, 956: 142, 957: 142, 928: 142,
    639: 134, 1055: 136, 640: 140, 641: 158, 642: 164, 643: 168, 645: 151, 646: 143, 649: 128,
    1051: 225, 1056: 148,
    # MoE / dropout / mixed precision
    611: 341, 345: 344, 1090: 344, 621: 237, 632: 55, 760: 273,
    # statistics
    793: 90, 795: 91,
    # RL
    # pre/post-norm questions share one group (defined in the normalization module)
}
ALIASES.update({
    1034: 531,
})
ALIASES.update({
    415: 355, 394: 326, 399: 706, 402: 395, 471: 470, 527: 528, 535: 528, 540: 528,
    583: 579, 598: 590, 592: 588, 517: 484, 919: 136, 920: 136, 929: 62, 931: 779, 932: 143, 933: 173,
    389: 384, 992: 384, 1009: 1010, 1057: 179, 1059: 468, 778: 209,
})
ALIASES.update({
    364: 721, 872: 422, 307: 306, 308: 561, 584: 306, 596: 589, 599: 1097, 658: 586, 673: 665, 674: 665, 683: 1083,
    939: 228, 1020: 859, 1028: 653,
})
