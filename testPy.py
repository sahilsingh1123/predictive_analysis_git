class Test():
    def __init__(self, varOne, varTwo):
        self.varOne = None if varOne == None else varOne
        self.varTwo = None if varTwo == None else varTwo

    def funcOne(self):
        varOneInsideFunc = self.varOne
        varOneInsideFunc = "inside func one"
        self.varOneInsideFunc = varOneInsideFunc
        varThreeInsideFunc = 23
        dic = {"keyOne": varOneInsideFunc, "keyTwo": varThreeInsideFunc}
        return dic

    def funcTwo(self):
        varTwoInsideFunc = self.varTwo
        var = self.funcOne()
        print(var['keyOne'])
        print(var['keyTwo'])
        keyImportedTwo=var['keyTwo']
        keyImported = var['keyOne']
        varTwoInsideFunc = "inside func two "+keyImported,keyImportedTwo
        print(varTwoInsideFunc)


classObj = Test(varOne=None, varTwo=None)
classObj.funcTwo()
